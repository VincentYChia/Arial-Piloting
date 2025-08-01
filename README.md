# Arial-Piloting: Edge-Native LLM Drone Control

**Fully autonomous drone control running entirely on consumer hardware with zero cloud dependencies.**

Built upon the [TypeFly](https://github.com/anyscale/typefly) architecture and the minispec langauge invented by Typefly, Arial-Piloting eliminates cloud dependencies while enhancing autonomous capabilities through a multi-model architecture. The system coordinates five specialized AI models to achieve autonomous flight planning, real-time replanning, and complex task execution on laptops with 8GB VRAM.

## 🚀 Key Features

- **Edge-Native Operation**: Complete elimination of cloud dependencies
- **Multi-Model Architecture**: 5 specialized AI models working in coordination
- **Autonomous Replanning**: Real-time task assessment and adaptation
- **Consumer Hardware**: Runs on standard laptops (8GB VRAM minimum)
- **Advanced Capabilities**: Person following, conditional logic, complex scene understanding

## 🎥 Demonstrations

- **Person Following**: 30-60 second continuous tracking in windy conditions
- **Complex Conditional Tasks**: "If you can see more than one chair behind you, then turn and go to the one with books on it"
- **Multi-Step Operations**: "Find something for me to eat. If you can, go for it and return. Otherwise, find something drinkable"

## 🏗️ Architecture

```
Natural Language Input → YoloV11 Vision → Qwen 3-4B Reasoning → Qwen 3-1.7B Code Generation → MiniSpec Execution → Replanning and Task Assesment 
                                     ↓
                               Auxiliary Systems:
                               • VLM Environment Probe  
                               • YoloE Scene Analysis
                               • Auto-Recentering & Replanning
```

### Core Components

- **Qwen 3-4B**: Flight plan reasoning and environmental analysis
- **Qwen 3-1.7B**: MiniSpec code generation and task assessment  
- **YoloV11**: Vision encoding and object detection
- **Qwen 2.5 VLM**: Environment probing for complex scenes
- **Yolo + LLM**: Abstract scene analysis

## 📋 Requirements

- **Hardware**: 8GB VRAM recommeneded 
- **[Ollama](https://ollama.ai)**: For local model serving
- **[YoloV11](https://github.com/ultralytics/ultralytics)**: Object detection
- **[YoloE](https://github.com/kadirnar/yolo-e)**: Enhanced scene analysis

Fine-tuned models and training data available in `/models` directory.

## 🤖 Model Configuration

All models use Q4_K_M quantization with identical inference parameters:

| Model | Purpose | Base Model | Quantization |
|-------|---------|------------|--------------|
| **Writing Model** | MiniSpec code generation | Qwen3-1.7B | Q4_K_M |
| **Reasoning Model** | Flight plan reasoning | Qwen3-4B | Q4_K_M |
| **Abstract Classifier** | Scene categorization | Qwen3-0.6B | Q4_K_M |
| **Replanning Model** | Task assessment | Qwen3-1.7B | Q4_K_M |
| **VLM** | Environment probing | Qwen2.5VL | Q4_K_M |

### Inference Parameters
- **top_k**: 20
- **top_p**: 0.95
- **repeat_penalty**: 1

## 📝 Citation

```bibtex
@misc{arial-piloting,
  title={Arial-Piloting: Edge-Native LLM Drone Piloting with Autonomous Reasoning},
  author={Vincent Y. Chia},
  year={2024},
  url={https://github.com/VincentYChia/Arial-Piloting}
}
```

**Note**: Research software. Follow local drone regulations and safety guidelines.
