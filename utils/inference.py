import torch
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

def load_model(model_path, device=None):
    """
    Load a trained FaceNet model from a .pt file
    
    Args:
        model_path: Path to the .pt model file
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        model: Loaded FaceNet model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize a new FaceNet model
    model = InceptionResnetV1(pretrained=None, classify=False).to(device)
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Set model to evaluation mode
    model.eval()
    
    print(f"Loaded model from {model_path} to {device}")
    return model

def preprocess_image(image_path, size=(160, 160)):
    """
    Preprocess an image for FaceNet
    
    Args:
        image_path: Path to the image file
        size: Size to resize the image to
        
    Returns:
        tensor: Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)  # Add batch dimension

def extract_embeddings(model, image_path, device=None):
    """
    Extract embeddings from an image using a trained FaceNet model
    
    Args:
        model: Trained FaceNet model
        image_path: Path to the image file
        device: Device to run inference on
        
    Returns:
        embeddings: Facial embeddings (512-dimensional vector)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Preprocess the image
    img_tensor = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # Extract embeddings
    with torch.no_grad():
        embeddings = model(img_tensor)
    
    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()

def compute_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        similarity: Cosine similarity score (higher means more similar)
    """
    return np.dot(embedding1, embedding2.T)[0][0]

def verify_faces(model, face1_path, face2_path, threshold=0.7, device=None):
    """
    Verify if two faces belong to the same person
    
    Args:
        model: Trained FaceNet model
        face1_path: Path to first face image
        face2_path: Path to second face image
        threshold: Similarity threshold (higher means stricter matching)
        device: Device to run inference on
        
    Returns:
        match: Boolean indicating if faces match
        similarity: Similarity score
    """
    # Extract embeddings
    embedding1 = extract_embeddings(model, face1_path, device)
    embedding2 = extract_embeddings(model, face2_path, device)
    
    # Compute similarity
    similarity = compute_similarity(embedding1, embedding2)
    
    # Determine if faces match
    match = similarity > threshold
    
    return match, similarity

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FaceNet inference utility")
    parser.add_argument('--model', type=str, required=True, help='Path to model .pt file')
    parser.add_argument('--image1', type=str, required=True, help='Path to first image')
    parser.add_argument('--image2', type=str, required=True, help='Path to second image')
    parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    
    # Load model
    model = load_model(args.model, device)
    
    # Verify faces
    match, similarity = verify_faces(model, args.image1, args.image2, args.threshold, device)
    
    # Print results
    print(f"Similarity score: {similarity:.4f}")
    print(f"Match: {match}")

if __name__ == "__main__":
    main()