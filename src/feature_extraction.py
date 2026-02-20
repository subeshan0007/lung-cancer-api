"""
Feature extraction from CT nodule patches.
Combines radiomics and deep learning features.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from typing import Dict, List, Tuple
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x  # fallback

# Maximum patch size for 3D CNN on 6GB GPU
# Patches larger than this will be downsampled before CNN input
CNN_PATCH_SIZE = (32, 64, 64)  # (D, H, W) — fits comfortably in 6GB VRAM
CNN_BATCH_SIZE = 2  # Small batch to prevent OOM on RTX 3060

# Optional radiomics import - only if installed and enabled
try:
    from radiomics import featureextractor
    RADIOMICS_AVAILABLE = True
except ImportError:
    RADIOMICS_AVAILABLE = False
    print("Warning: pyradiomics not installed. Radiomics features will be disabled.")


class RadiomicsFeatureExtractor:
    """Extract radiomics features using PyRadiomics."""
    
    def __init__(self, config: Dict):
        if not RADIOMICS_AVAILABLE:
            raise ImportError("pyradiomics not installed")
        self.config = config
        self.extractor = featureextractor.RadiomicsFeatureExtractor()
        
        # Configure extractor
        self.extractor.enableAllFeatures()
        
    def extract(self, patch: np.ndarray) -> np.ndarray:
        """
        Extract radiomics features from 3D patch.
        
        Args:
            patch: 3D numpy array (D, H, W)
            
        Returns:
            1D feature vector
        """
        try:
            # Convert to SimpleITK image
            image = sitk.GetImageFromArray(patch.astype(np.float32))
            
            # Create simple mask (all ones)
            mask = sitk.GetImageFromArray(np.ones_like(patch, dtype=np.uint8))
            
            # Extract features
            features = self.extractor.execute(image, mask)
            
            # Filter numerical features only
            feature_vector = []
            for key, value in features.items():
                if isinstance(value, (int, float, np.number)):
                    feature_vector.append(float(value))
            
            return np.array(feature_vector, dtype=np.float32)
            
        except Exception as e:
            print(f"Error extracting radiomics features: {e}")
            # Return zeros as fallback
            return np.zeros(100, dtype=np.float32)


class ResNet3D(nn.Module):
    """Simple 3D ResNet for feature extraction."""
    
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Encoder path
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(2)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(256, feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from 3D patch.
        
        Args:
            x: (batch_size, 1, D, H, W)
            
        Returns:
            (batch_size, feature_dim)
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # FC layer
        x = self.fc(x)
        
        return x


class DeepLearningFeatureExtractor:
    """Extract deep learning features from 3D CNN."""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        
        feature_dim = config['features']['deep_learning']['feature_dim']
        self.model = ResNet3D(feature_dim=feature_dim).to(device)
        self.model.eval()
        
    def extract_batch(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Extract features from batch of patches.
        Automatically downsamples large patches to fit in GPU memory.
        
        Args:
            patches: (batch_size, 1, D, H, W)
            
        Returns:
            (batch_size, feature_dim)
        """
        with torch.no_grad():
            # Downsample large patches to CNN_PATCH_SIZE to prevent OOM
            _, _, d, h, w = patches.shape
            td, th, tw = CNN_PATCH_SIZE
            if d > td or h > th or w > tw:
                patches = F.interpolate(
                    patches, size=CNN_PATCH_SIZE, mode='trilinear', align_corners=False
                )
            
            # Process in small sub-batches to stay within VRAM
            all_features = []
            for start in range(0, len(patches), CNN_BATCH_SIZE):
                mini_batch = patches[start:start + CNN_BATCH_SIZE].to(self.device)
                feats = self.model(mini_batch)
                all_features.append(feats.cpu())
                del mini_batch, feats
                torch.cuda.empty_cache()
            
            features = torch.cat(all_features, dim=0)
        return features
    
    def save(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def extract_features_from_samples(samples: List[Dict], config: Dict, 
                                   device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract combined features from all samples.
    
    Args:
        samples: List of preprocessed samples
        config: Configuration dictionary
        device: Computing device
        
    Returns:
        features: (num_samples, feature_dim) numpy array
        labels: (num_samples,) numpy array
    """
    use_radiomics = config['features']['radiomics']['enabled'] and RADIOMICS_AVAILABLE
    use_dl = config['features']['deep_learning']['enabled']
    
    if use_radiomics and not RADIOMICS_AVAILABLE:
        print("WARNING: Radiomics enabled in config but pyradiomics not installed. Using deep learning features only.")
        use_radiomics = False
    
    all_features = []
    all_labels = []
    
    # Initialize extractors
    if use_radiomics:
        radiomics_extractor = RadiomicsFeatureExtractor(config)
    
    if use_dl:
        dl_extractor = DeepLearningFeatureExtractor(config, device)
        batch_size = config['training']['batch_size']
    
    print(f"Extracting features from {len(samples)} samples...")
    print(f"  Radiomics: {'enabled' if use_radiomics else 'disabled'}")
    print(f"  Deep Learning: {'enabled' if use_dl else 'disabled'}")
    
    for i in tqdm(range(0, len(samples), batch_size if use_dl else 1), desc="Feature extraction"):
        batch_samples = samples[i:i+batch_size] if use_dl else [samples[i]]
        
        batch_features = []
        
        # Extract radiomics features
        if use_radiomics:
            for sample in batch_samples:
                radiomics_feats = radiomics_extractor.extract(sample['nodule_patch'])
                batch_features.append(radiomics_feats)
        
        # Extract deep learning features
        if use_dl:
            patches = torch.stack([
                torch.from_numpy(s['nodule_patch'][np.newaxis, ...].astype(np.float32))
                for s in batch_samples
            ])
            dl_feats = dl_extractor.extract_batch(patches).numpy()
            del patches  # Free CPU tensor immediately
            gc.collect()
            
            if use_radiomics:
                # Concatenate radiomics and DL features
                batch_features = [
                    np.concatenate([rad_f, dl_f])
                    for rad_f, dl_f in zip(batch_features, dl_feats)
                ]
            else:
                batch_features = dl_feats
        
        all_features.extend(batch_features)
        all_labels.extend([s['label'] for s in batch_samples])
    
    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return features, labels


class FeatureExtractor:
    """Wrapper class for feature extraction during inference."""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.use_radiomics = config['features']['radiomics']['enabled'] and RADIOMICS_AVAILABLE
        self.use_dl = config['features']['deep_learning']['enabled']
        
        # Initialize extractors
        if self.use_radiomics:
            self.radiomics_extractor = RadiomicsFeatureExtractor(config)
        
        if self.use_dl:
            self.dl_extractor = DeepLearningFeatureExtractor(config, device)
    
    def extract_single(self, patch: np.ndarray) -> np.ndarray:
        """
        Extract features from a single 3D patch.
        
        Args:
            patch: 3D numpy array (D, H, W) or (1, D, H, W)
            
        Returns:
            1D feature vector
        """
        # Remove batch dimension if present
        if len(patch.shape) == 4:
            patch = patch[0]
        
        features = []
        
        # Extract radiomics features
        if self.use_radiomics:
            radiomics_feats = self.radiomics_extractor.extract(patch)
            features.append(radiomics_feats)
        
        # Extract deep learning features
        if self.use_dl:
            patch_tensor = torch.from_numpy(
                patch[np.newaxis, np.newaxis, ...].astype(np.float32)
            )
            dl_feats = self.dl_extractor.extract_batch(patch_tensor).numpy()[0]
            features.append(dl_feats)
        
        # Concatenate all features
        if len(features) > 1:
            return np.concatenate(features)
        else:
            return features[0]


# ==============================================================================
# 2D IMAGE SUPPORT
# ==============================================================================

class ResNet2D(nn.Module):
    """
    2D ResNet feature extractor using pretrained torchvision model.
    Extracts 512-dimensional features from 2D images (JPG/PNG/DICOM slices).
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision.models as models
        
        # Load pretrained ResNet18
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        
        # Remove final FC layer — use avgpool output as features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512
        
        # Freeze pretrained weights
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.eval()
        print(f"Initialized ResNet2D feature extractor (pretrained={pretrained})")
        print(f"  Output features: {self.feature_dim}")
    
    def forward(self, x):
        """
        Extract features from 2D images.
        
        Args:
            x: Tensor of shape (batch, 3, H, W)
            
        Returns:
            Features of shape (batch, 512)
        """
        with torch.no_grad():
            features = self.features(x)
            features = features.squeeze(-1).squeeze(-1)
        return features


class DeepLearningFeatureExtractor2D:
    """
    Wrapper for 2D feature extraction with GPU support.
    Handles image preprocessing, batching, and GPU memory management.
    """
    
    def __init__(self, device='cpu', batch_size=16):
        self.device = device
        self.batch_size = batch_size
        self.model = ResNet2D(pretrained=True).to(device)
        self.model.eval()
    
    def preprocess_2d_image(self, image):
        """
        Preprocess a 2D image for ResNet2D.
        
        Args:
            image: numpy array — (H, W), (H, W, 1), (H, W, 3), or (1, H, W)
            
        Returns:
            Tensor of shape (1, 3, 224, 224) ready for ResNet2D
        """
        # Handle different input shapes
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image[0]  # (1, H, W) → (H, W)
            elif image.shape[2] == 1:
                image = image[:, :, 0]  # (H, W, 1) → (H, W)
            elif image.shape[2] == 3:
                # RGB → grayscale
                image = np.mean(image, axis=2)
        
        # Now image is (H, W)
        # Resize to 224x224 for ResNet
        # Use .tolist() to bypass numpy↔PyTorch version incompatibilities
        tensor = torch.tensor(image.tolist(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize to [0, 1] if not already
        if tensor.max() > 1.0:
            tensor = tensor / tensor.max()
        
        # Repeat grayscale to 3 channels (ResNet expects RGB)
        tensor = tensor.repeat(1, 3, 1, 1)
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor
    
    def extract_single(self, image):
        """
        Extract features from a single 2D image.
        
        Args:
            image: numpy array (H, W) or (H, W, C)
            
        Returns:
            Feature vector of shape (512,)
        """
        tensor = self.preprocess_2d_image(image).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
        result = features.cpu().numpy()[0]
        
        # Cleanup
        del tensor, features
        if self.device != 'cpu':
            torch.cuda.empty_cache()
        
        return result
    
    def extract_batch(self, images):
        """
        Extract features from a batch of 2D images.
        
        Args:
            images: List of numpy arrays (H, W) or (H, W, C)
            
        Returns:
            Feature array of shape (n_images, 512)
        """
        all_features = []
        
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            tensors = [self.preprocess_2d_image(img) for img in batch]
            batch_tensor = torch.cat(tensors, dim=0).to(self.device)
            
            with torch.no_grad():
                features = self.model(batch_tensor)
            
            all_features.append(features.cpu().numpy())
            
            del batch_tensor, features
            if self.device != 'cpu':
                torch.cuda.empty_cache()
        
        return np.concatenate(all_features, axis=0)


def load_2d_image(image_path):
    """
    Load a 2D image from various formats.
    
    Args:
        image_path: Path to image file (.jpg, .png, .jpeg, .dcm)
        
    Returns:
        numpy array (H, W) normalized to [0, 1]
    """
    from pathlib import Path
    
    path = Path(image_path)
    ext = path.suffix.lower()
    
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        from PIL import Image
        img = Image.open(str(image_path)).convert('L')  # Convert to grayscale
        image = np.array(img, dtype=np.float32) / 255.0
        
    elif ext == '.dcm':
        # Prefer SimpleITK, but fall back to pydicom if needed
        loaded = False
        if HAS_SITK:
            try:
                sitk_image = sitk.ReadImage(str(image_path))
                image = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
                loaded = True
            except Exception as e:
                print(f"Warning: SimpleITK failed to load DICOM ('{e}'). Trying pydicom.")
        if not loaded:
            try:
                import pydicom
                ds = pydicom.dcmread(str(image_path))
                image = ds.pixel_array.astype(np.float32)
            except ImportError:
                raise ImportError("Neither SimpleITK nor pydicom available for DICOM loading")

        # If 3D with multiple slices, take middle slice
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]
        elif image.ndim == 3:
            mid = image.shape[0] // 2
            image = image[mid]

        # Normalize to [0, 1]
        if image.max() != image.min():
            image = (image - image.min()) / (image.max() - image.min())
    
    elif ext == '.npy':
        image = np.load(str(image_path)).astype(np.float32)
        if image.ndim == 3:
            mid = image.shape[0] // 2
            image = image[mid]
        if image.max() != image.min():
            image = (image - image.min()) / (image.max() - image.min())
    
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .jpg, .png, .jpeg, .dcm, or .npy")
    
    return image

