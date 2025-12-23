#!/usr/bin/env python3
"""
Final Comprehensive Round-Trip Test for vLLM Embedding Decoder

Pipeline: text → Vello render (640×640) → vision encoder → vLLM decoder → text

Tests the complete encoder-decoder separation with configurable text length.
Saves test images and ground truth to outputs/test_images/ with timestamp.
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from PIL import Image
import sys

# Add parent directories to path
ocrinfer_root = Path(__file__).parent.parent
sys.path.insert(0, str(ocrinfer_root.resolve()))
sys.path.insert(0, str(ocrinfer_root.parent.resolve()))  # For OCRFlow and Renderer

from OCRInfer.decoder import VLLMEmbeddingDecoder
from OCRInfer.encoder import DPSKOCREncoder
from OCRInfer.config import MODEL_PATH

# Import Vello renderer
try:
    from Renderer import VelloRenderer
    VELLO_AVAILABLE = True
except ImportError as e:
    print(f"❌ Vello renderer not available: {e}")
    print("Please build Vello renderer:")
    print("  cd Renderer")
    print("  maturin develop --release")
    VELLO_AVAILABLE = False


def generate_test_text(target_tokens: int = 1000) -> str:
    """
    Generate diverse test text with EXACTLY target_tokens (guaranteed).

    Creates unique, non-repetitive content covering multiple domains:
    OCR, machine learning, document processing, vision systems, and NLP.

    Args:
        target_tokens: Exact number of tokens (words) to generate

    Returns:
        Generated text string with exactly target_tokens words
    """
    # Extended diverse content blocks to have enough content for 1000+ tokens
    paragraphs = [
        # Para 1: OCR history and evolution (expanded)
        """Optical Character Recognition (OCR) technology has fundamentally transformed how we interact with textual information in the digital age. From its humble beginnings in the early 20th century, when primitive machines could barely recognize a few standardized characters, OCR has evolved into a sophisticated field combining computer vision, machine learning, natural language processing, and computational linguistics. Today's OCR systems can handle an astounding variety of text types, from pristine printed documents to degraded historical manuscripts, from perfectly aligned text to skewed images captured on mobile phones. The technological journey reflects decades of innovation across multiple disciplines.""",

        # Para 2: OCR evolution and AI advances (expanded)
        """The evolution of OCR reflects broader advances in artificial intelligence and pattern recognition. Early template-matching approaches worked adequately for standard fonts but failed catastrophically with variations and deformations. Feature-based methods extracted geometric and topological properties from character shapes and learned discriminative patterns. Modern deep learning systems employ end-to-end trainable models that learn optimal representations automatically, eliminating manual feature engineering and dramatically improving robustness across diverse documents, scripts, and degradation conditions. Statistical methods now combine with neural approaches.""",

        # Para 3: Deep learning architectures (expanded)
        """Deep learning architectures have revolutionized OCR performance through sophisticated neural network designs and training methodologies. Convolutional Neural Networks excel at capturing spatial hierarchies in visual data through local receptive fields and feature maps. Recurrent Neural Networks and Transformers model sequential dependencies in text, enabling context-aware predictions and long-range understanding. Vision transformers use self-attention mechanisms to recognize global patterns and long-range dependencies across entire images, fundamentally changing how we approach visual understanding. Hybrid architectures combine strengths.""",

        # Para 4: Preprocessing pipelines (expanded)
        """Preprocessing pipelines form the critical foundation of robust OCR systems and quality output. Text deskewing corrects rotated documents to horizontal orientation using geometric algorithms and line detection. Binarization converts grayscale images into pure black and white, simplifying subsequent processing and reducing complexity. Denoising removes artifacts, salt-and-pepper noise, and artifacts introduced during scanning or photography. Layout analysis identifies distinct regions: paragraphs, tables, figures, captions, headers, and footers with spatial reasoning. Document segmentation properly interprets complex multi-column layouts and reading order.""",

        # Para 5: Computer vision techniques (expanded)
        """Computer vision techniques for document understanding extend far beyond simple character recognition into complex scene analysis. Line detection identifies text baselines and boundaries using edge detection and Hough transforms. Text orientation estimation determines reading direction whether horizontal, vertical, or rotated. Table structure recognition identifies cells, rows, columns, and relationships between elements. Figure detection separates illustrations from text using content analysis. Signature identification enables document authentication and verification. Scene text detection handles natural images with curved text, artistic fonts, shadows, and occlusion in varying lighting.""",

        # Para 6: Language models (expanded)
        """Language models complement visual recognition by leveraging linguistic knowledge, contextual information, and semantic understanding. Statistical models predict probable character sequences based on language patterns, grammar, and statistical regularities learned from corpus data. Post-processing integrates visual and linguistic knowledge to resolve ambiguities and improve consistency. Confidence scores indicate recognition certainty and allow selective human review. Error correction identifies and fixes common OCR mistakes from confusion between similar characters and systematic biases. Context windows use surrounding words to disambiguate questionable characters.""",

        # Para 7: Multilingual OCR (expanded)
        """Multilingual OCR presents unique challenges and opportunities in our interconnected globalized world. Different scripts have different writing systems: logographic (Chinese, Japanese, Korean), alphabetic (Latin, Cyrillic, Greek, Arabic), and abugida (Devanagari, Thai, Khmer). Character sets vary dramatically in size and complexity from 26 Latin letters to thousands of Chinese characters. Font families differ across cultures with region-specific typographic conventions. Historical documents use archaic scripts and typography from earlier centuries. Machine learning models must be trained on diverse multilingual datasets to achieve acceptable accuracy.""",

        # Para 8: Benchmarking metrics (expanded)
        """Benchmarking OCR systems requires comprehensive evaluation metrics beyond simple character accuracy for meaningful assessment. Character error rate measures individual character mistakes including substitutions, insertions, and deletions. Word error rate captures word-level performance considering only complete word correctness. Document-level metrics assess overall quality for real-world applications and business impact. Confidence-weighted metrics penalize high-confidence errors more heavily than uncertain mistakes. Edit distance metrics measure minimum edits required to transform output to ground truth. Context-aware metrics evaluate semantic preservation and meaning maintenance, not just character matching.""",

        # Para 9: Evaluation datasets (expanded)
        """Evaluation datasets for OCR encompass diverse document types reflecting real-world variability and challenges. Historical manuscript datasets contain aged paper with fading ink, water damage, and deterioration from centuries. Modern printed documents use clean typography with standard fonts and high quality printing. Handwritten text datasets capture natural writing variation, individual slant, cursiveness, and personal writing styles. Scene text datasets include photographs with natural lighting, shadows, occlusion, perspective distortion, and complex backgrounds. Synthetic datasets enable controlled testing of specific challenges like noise, blur, rotation, and degradation.""",

        # Para 10: Industry applications (expanded)
        """Applications of OCR technology extend throughout modern society across numerous industries and use cases. Document digitization converts physical archives into searchable digital libraries enabling preservation and access. Accessibility technology converts printed materials into screen-reader-compatible formats benefiting visually impaired users. Payment processing recognizes bank account numbers, expiration dates, and security codes for financial transactions. Postal automation sorts mail based on address recognition enabling efficient delivery logistics. License plate recognition enables traffic monitoring, toll collection, and law enforcement applications. Form processing automatically extracts structured data from filled questionnaires, surveys, and administrative documents.""",

        # Para 11: Future directions (expanded more)
        """Future developments in OCR technology will likely focus on end-to-end learning systems, real-time processing, and seamless integration with other emerging AI technologies. Cross-modal learning combining vision and language models will substantially improve contextual understanding and semantic accuracy. Edge computing will enable local processing without cloud dependency or network latency. Few-shot and zero-shot learning will significantly reduce annotation requirements and training data needs. Integration with knowledge graphs will enhance semantic understanding beyond simple character recognition and enable reasoning. Video OCR will extract and process text from moving images and streaming video content in real-time. Handwriting recognition will eventually achieve performance comparable to printed text recognition through advanced deep learning.""",

        # Para 12: Practical considerations
        """Practical considerations for OCR deployment include system scalability, computational efficiency, and user experience design. Cloud-based OCR services provide flexibility and easy integration but raise privacy and latency concerns. On-premise solutions offer better data control and reduced network dependency but require infrastructure investment. Hybrid approaches balance benefits and drawbacks by processing sensitive data locally and leveraging cloud resources for scaling. Cost optimization involves selecting appropriate hardware, choosing between GPU and CPU acceleration, and implementing efficient caching strategies. Quality assurance requires comprehensive testing across diverse inputs and systematic monitoring of production performance metrics.""",

        # Para 13: Integration with modern systems
        """Integration of OCR technology with modern software systems involves standardized APIs, proper error handling, and graceful degradation strategies. RESTful web services enable easy integration with diverse applications from mobile apps to enterprise systems. Database integration allows storing extracted text with original images for audit trails and reprocessing. Workflow automation connects OCR with downstream processing like data entry validation and document classification. APIs must handle edge cases including empty documents, corrupted files, and unsupported languages with meaningful error messages. Monitoring systems track performance metrics like throughput, latency, and accuracy to identify degradation early.""",

        # Para 14: Security and privacy considerations
        """Security and privacy considerations are paramount in OCR system deployment, especially handling sensitive documents. Data encryption protects text during transmission and storage using industry-standard protocols and algorithms. Access control mechanisms ensure only authorized users can view sensitive extracted content. Compliance with regulations like GDPR, HIPAA, and PCI-DSS requires careful handling of personal health information and financial data. Audit logging tracks who accessed what information and when for accountability. Secure deletion ensures processed data doesn't persist longer than necessary. Privacy-preserving techniques like federated learning enable model improvement without centralizing sensitive data.""",

        # Para 15: Performance optimization strategies
        """Performance optimization strategies significantly impact OCR system efficiency and user experience. Model quantization reduces neural network size and memory requirements while maintaining accuracy through int8 and mixed-precision techniques. Batch processing increases throughput by processing multiple documents simultaneously on GPU. Caching mechanisms store frequently accessed results to avoid redundant computation. Asynchronous processing prevents UI blocking when handling time-consuming OCR tasks. Progressive rendering shows partial results as they become available. Load balancing distributes requests across multiple servers for horizontal scalability. Resource monitoring identifies bottlenecks enabling targeted optimization efforts.""",

        # Para 16: Advanced OCR techniques and research
        """Advanced OCR techniques push the boundaries of what's possible in text recognition and understanding. End-to-end learnable architectures eliminate intermediate steps like character segmentation. Weakly-supervised learning enables training with incomplete or imperfect annotations. Active learning strategically selects which samples to annotate for maximum learning impact. Domain adaptation transfers knowledge from synthetic data to real documents. Curriculum learning arranges training samples from easy to hard for better convergence. Ensemble methods combine multiple models for improved accuracy and robustness. Transfer learning leverages pre-trained models from related tasks reducing training requirements and time.""",

        # Para 17: Real-world deployment challenges
        """Real-world deployment challenges present obstacles beyond theoretical OCR accuracy improvements. Legacy document formats require support for obsolete technologies and standards. Document aging and deterioration present variable challenges across historical collections. Varying lighting conditions in document photography affect image quality substantially. Printer variability across decades and manufacturers creates style differences. User expectations management ensures stakeholders understand accuracy limitations realistically. Cost-benefit analysis balances OCR investment against manual transcription expenses. Training data collection and annotation require significant time and financial resources. Model maintenance ensures systems continue performing as data distributions shift over time.""",

        # Para 18: Data annotation and labeling strategies
        """Data annotation and labeling form the foundation of supervised learning for OCR systems. Expert human annotators must carefully transcribe text from images with pixel-level accuracy for training data. Quality control mechanisms ensure consistency across multiple annotators through inter-annotator agreement metrics. Crowdsourcing platforms enable large-scale annotation but require careful task design and worker qualification. Active learning reduces annotation burden by identifying samples most useful for model improvement. Weak supervision leverages noisy labels from automatic systems with lower cost. Multi-modal annotation captures bounding boxes, character segmentation, and confidence levels. Version control systems manage evolving annotation guidelines and corrections throughout campaigns.""",

        # Para 19: Emerging technologies and research frontiers
        """Emerging technologies continue pushing OCR capabilities into new domains and applications. Generative models like diffusion networks can synthesize realistic document variations for augmentation. Vision-language models combine OCR with semantic understanding for context-aware extraction. Self-supervised learning methods learn visual representations without manual labeling. Contrastive learning improves model robustness through diverse augmentation strategies. Neural architecture search automatically discovers optimal network designs for OCR tasks. Graph neural networks model relationships between text regions and document structure. Quantum computing promises exponential speedups for certain pattern matching algorithms in future systems.""",

        # Para 20: Cost-benefit analysis and ROI considerations
        """Cost-benefit analysis requires careful evaluation of OCR implementation economics and business impact. Infrastructure costs include hardware procurement, maintenance, and power consumption for GPU clusters. Software licensing covers proprietary systems, cloud services, and open-source support contracts. Personnel expenses include skilled engineers, data scientists, and annotation workforce. Training costs encompass model development, dataset creation, and validation. Operational costs include monitoring, debugging, and continuous improvement efforts. Benefits calculation measures productivity gains from automation, error reduction, and processing speed improvements. Return on investment timelines vary from months for simple automation to years for complex systems. Total cost of ownership analysis informs technology platform selection decisions.""",

        # Para 21: Document management system integration
        """Integration of OCR with document management systems enables end-to-end digital workflow transformation. Indexing extracted text enables full-text search across document archives and repositories. Metadata extraction captures document properties like date, author, and classification automatically. Workflow automation triggers subsequent processing steps based on document content and classification. Compliance features ensure proper handling of sensitive information and audit trails. Version control tracks document modifications and maintains historical records. Integration with business intelligence systems enables analytics on document collections. Backup and disaster recovery systems protect OCR results and prevent data loss during system failures.""",

        # Para 22: User experience design for OCR applications
        """User experience design for OCR applications must balance automation with human oversight and control. Confidence indicators show users which parts of the extraction are most reliable and which need review. Progressive disclosure reveals detailed information only when needed by power users. Undo and correction mechanisms allow users to fix errors and improve future predictions. Keyboard shortcuts and batch operations increase productivity for high-volume processing scenarios. Accessibility features ensure visually impaired users can navigate and review OCR results. Responsive design adapts to different screen sizes and devices from mobile to desktop. Real-time feedback shows processing progress and estimated completion times accurately.""",
    ]

    # Build complete text first (preserve paragraph structure)
    all_text = '\n\n'.join(paragraphs)
    all_words = all_text.split()
    all_tokens = len(all_words)

    if all_tokens >= target_tokens:
        # We have enough content, extract exactly target_tokens words
        # But preserve paragraph breaks by working paragraph by paragraph
        result_paragraphs = []
        words_collected = 0

        for para in paragraphs:
            para_words = para.split()
            if words_collected + len(para_words) <= target_tokens:
                # Add full paragraph
                result_paragraphs.append(para)
                words_collected += len(para_words)
            else:
                # Add partial paragraph to reach target
                remaining_words = target_tokens - words_collected
                partial_para = ' '.join(para_words[:remaining_words])
                result_paragraphs.append(partial_para)
                break

        return '\n\n'.join(result_paragraphs)
    else:
        # Not enough content even with all paragraphs
        # This shouldn't happen with our 11 paragraphs, but handle it gracefully
        # Repeat content cyclically to reach target
        words_needed = target_tokens
        result_words = []
        para_index = 0

        while len(result_words) < words_needed:
            para_words = paragraphs[para_index % len(paragraphs)].split()
            words_to_add = min(len(para_words), words_needed - len(result_words))
            result_words.extend(para_words[:words_to_add])
            para_index += 1

        return ' '.join(result_words[:target_tokens])


def render_text_with_vello(text: str, width: int = 640, height: int = 640) -> Image.Image:
    """
    Render text using Vello GPU renderer.

    Args:
        text: Text to render
        width: Image width
        height: Image height

    Returns:
        PIL Image
    """
    if not VELLO_AVAILABLE:
        raise RuntimeError("Vello renderer not available")

    # Create renderer with proven working settings
    renderer = VelloRenderer(
        width=width,
        height=height,
        padding=20,
        min_font_size=5.0,
        max_font_size=20.0
    )

    # Render text (returns list of numpy arrays)
    images = renderer.render_batch([text])

    # Convert to PIL Image
    img_array = images[0]  # Shape: (640, 640, 3), dtype: uint8
    img = Image.fromarray(img_array, mode='RGB')

    return img


def save_test_artifacts(image: Image.Image, ground_truth: str, output_dir: Path, timestamp: str):
    """
    Save test image and ground truth to outputs directory.

    Args:
        image: Rendered image
        ground_truth: Ground truth text
        output_dir: Output directory path
        timestamp: Timestamp string for filenames

    Returns:
        Path to the saved ground truth file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save image
    img_path = output_dir / f"roundtrip_test_{timestamp}.png"
    image.save(img_path)
    print(f"  💾 Saved image: {img_path}")

    # Save ground truth (.mmd format)
    gt_path = output_dir / f"roundtrip_test_{timestamp}.mmd"
    with open(gt_path, 'w', encoding='utf-8') as f:
        f.write(ground_truth)
    print(f"  💾 Saved ground truth: {gt_path}")

    return gt_path


def calculate_metrics(ground_truth: str, generated: str):
    """
    Calculate meaningful OCR accuracy metrics (without misleading character-level matching).

    Args:
        ground_truth: Original text
        generated: Generated text

    Returns:
        Dictionary of metrics
    """
    # Word-level accuracy (more meaningful for OCR)
    gt_words = ground_truth.split()
    gen_words = generated.split()
    word_accuracy = SequenceMatcher(None, gt_words, gen_words).ratio()

    # Token/word counts
    gt_tokens = len(gt_words)
    gen_tokens = len(gen_words)

    # Vocabulary retention (what % of unique words were recovered)
    gt_word_set = set(w.lower() for w in gt_words)
    gen_word_set = set(w.lower() for w in gen_words)
    vocab_overlap = len(gt_word_set & gen_word_set) / len(gt_word_set) if gt_word_set else 0.0

    return {
        'word_accuracy': word_accuracy,
        'gt_chars': len(ground_truth),
        'gen_chars': len(generated),
        'gt_tokens': gt_tokens,
        'gen_tokens': gen_tokens,
        'token_recovery_rate': gen_tokens / gt_tokens if gt_tokens > 0 else 0.0,
        'vocab_overlap': vocab_overlap
    }


def main(target_tokens: int = 900):
    """
    Run complete round-trip test.

    Args:
        target_tokens: Target number of tokens for test text (max: 900 for 640×640)

    Returns:
        Exit code (0 for pass, 1 for fail)
    """
    print("="*80)
    print("Final Comprehensive Round-Trip Test")
    print("="*80)
    print(f"Target tokens: {target_tokens}")
    print(f"Pipeline: text → Vello (640×640) → encoder → decoder → text")
    print()

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup output directory (OCRInfer/outputs/test_images)
    output_dir = Path(__file__).parent.parent / "outputs" / "test_images"

    # Step 1: Generate test text
    print("[Step 1/5] Generating test text...")
    ground_truth = generate_test_text(target_tokens=target_tokens)
    actual_tokens = len(ground_truth.split())
    print(f"  ✓ Generated {actual_tokens} tokens ({len(ground_truth)} chars)")
    print(f"  Preview: {ground_truth[:200]}...")
    print()

    # Step 2: Render with Vello
    print("[Step 2/5] Rendering text with Vello GPU renderer...")
    if not VELLO_AVAILABLE:
        print("  ❌ Vello not available, cannot continue")
        return 1

    image = render_text_with_vello(ground_truth, width=640, height=640)
    print(f"  ✓ Rendered to 640×640 image")

    # Save artifacts
    gt_path = save_test_artifacts(image, ground_truth, output_dir, timestamp)
    print()

    # Step 3: Encode with vision encoder
    print("[Step 3/5] Encoding image to visual tokens...")
    encoder = DPSKOCREncoder(
        model_path=MODEL_PATH,
        device='cuda',
        dtype=torch.bfloat16
    )

    vistoks = encoder.encode_images(
        images=[image],
        return_global=False,
        return_local=True
    )[0]

    print(f"  ✓ Visual tokens: {vistoks.shape}")
    print(f"  📊 Token stats: min={vistoks.min():.4f}, max={vistoks.max():.4f}, mean={vistoks.mean():.4f}, std={vistoks.std():.4f}")

    # Clean up encoder
    del encoder
    torch.cuda.empty_cache()
    print()

    # Step 4: Decode with vLLM
    print("[Step 4/5] Decoding with vLLM embedding decoder...")
    decoder = VLLMEmbeddingDecoder(
        model_path=MODEL_PATH,
        gpu_memory_utilization=0.7,
        max_model_len=8192,
        dtype='bfloat16'
    )

    generated_text = decoder.decode(
        visual_embeddings=vistoks,
        prompt="Free OCR.",
        max_tokens=target_tokens + 200,  # Allow some buffer for formatting
        temperature=0.0,
        ngram_size=30,
        window_size=90
    )

    print(f"  ✓ Generated {len(generated_text)} chars")

    # Save generated output
    generated_path = str(gt_path).replace('.mmd', '_generated.txt')
    with open(generated_path, 'w', encoding='utf-8') as f:
        f.write(generated_text)
    print(f"  💾 Saved generated: {generated_path}")
    print()

    # Step 5: Validate accuracy
    print("[Step 5/5] Validating round-trip accuracy...")
    metrics = calculate_metrics(ground_truth, generated_text)

    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"\n📊 Metrics:")
    print(f"  Ground truth:     {metrics['gt_tokens']:5d} tokens, {metrics['gt_chars']:5d} chars")
    print(f"  Generated:        {metrics['gen_tokens']:5d} tokens, {metrics['gen_chars']:5d} chars")
    print(f"  Word accuracy:    {metrics['word_accuracy']*100:5.2f}%")
    print(f"  Token recovery:   {metrics['token_recovery_rate']*100:5.2f}%")
    print(f"  Vocabulary:       {metrics['vocab_overlap']*100:5.2f}% words retained")

    # Show sample comparison
    print(f"\n📝 Ground Truth (first 300 chars):")
    print("-"*80)
    print(ground_truth[:300])
    print()

    print(f"📝 Generated (first 300 chars):")
    print("-"*80)
    print(generated_text[:300])
    print()

    # Determine pass/fail based on meaningful metrics
    word_threshold = 0.50      # 50% word accuracy
    token_threshold = 0.70     # 70% token recovery
    vocab_threshold = 0.60     # 60% vocabulary retention

    if (metrics['word_accuracy'] >= word_threshold and
        metrics['token_recovery_rate'] >= token_threshold and
        metrics['vocab_overlap'] >= vocab_threshold):
        print("="*80)
        print("✅ PASS: Round-trip test successful!")
        print(f"   Word accuracy:       {metrics['word_accuracy']*100:.1f}% (threshold: {word_threshold*100:.0f}%)")
        print(f"   Token recovery:      {metrics['token_recovery_rate']*100:.1f}% (threshold: {token_threshold*100:.0f}%)")
        print(f"   Vocabulary retained: {metrics['vocab_overlap']*100:.1f}% (threshold: {vocab_threshold*100:.0f}%)")
        print("="*80)
        return 0
    else:
        print("="*80)
        print("❌ FAIL: Round-trip accuracy below threshold")
        print(f"   Word accuracy:       {metrics['word_accuracy']*100:.1f}% (threshold: {word_threshold*100:.0f}%)")
        print(f"   Token recovery:      {metrics['token_recovery_rate']*100:.1f}% (threshold: {token_threshold*100:.0f}%)")
        print(f"   Vocabulary retained: {metrics['vocab_overlap']*100:.1f}% (threshold: {vocab_threshold*100:.0f}%)")
        print("="*80)
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Round-trip accuracy test")
    parser.add_argument(
        "--tokens",
        type=int,
        default=900,
        help="Target number of tokens for test text (default: 900, max: 900 for 640×640 image)"
    )
    args = parser.parse_args()

    # Enforce 900-token limit for 640×640 images
    if args.tokens > 900:
        print("="*80)
        print("⚠️  WARNING: Token limit exceeded!")
        print("="*80)
        print(f"\nRequested: {args.tokens} tokens")
        print(f"Maximum:   900 tokens (conservative limit for 640×640 image)")
        print(f"\nReasoning:")
        print(f"  - 900 tokens ≈ 7425 characters")
        print(f"  - Renders at ~6.5pt font size")
        print(f"  - Balances content density with OCR accuracy")
        print("\nAdjusting to 900 tokens...\n")
        args.tokens = 900

    exit_code = main(target_tokens=args.tokens)
    sys.exit(exit_code)
