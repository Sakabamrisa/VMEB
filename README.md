# VMEB: Vid-LLMs Model Editing Benchmark

**VMEB** is a benchmark designed to evaluate model editing capabilities in Video-based Large Language Models (Vid-LLMs), focusing on Reliability, Locality, Generality, and Robustness.

## Acknowledgements
This project is modified based on **VLKEB**.
- **VLKEB Repository**: [[https://github.com/VLKEB/VLKEB](https://github.com/VLKEB/VLKEB)]

## Environment & Setup
For environment configuration, installation, and detailed usage instructions, please refer to the **VLKEB** repository linked above. The setup process is compatible with this codebase.

## Run
To run the editing experiments, use the following command:

```bash
python multimodal_edit.py [FUNC_NAME] [HOP_NUM]
```

> **Note**: Please refer to `multimodal_edit.py` to check the available `[FUNC_NAME]` options. To train the model, please adjust hparams following the configurations in the appendix.