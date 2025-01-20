# Quick Promptfoo Evaluation Guide

## Quick Start

1. **Set API Keys:** Ensure required API keys (e.g.`ANTHROPIC_API_KEY`) are set in your environment. If using a `.env` file, you can load it:
   ```bash
   export $(cat ../../.env | xargs)
   ```

2. **Run Evaluations:** Use the following commands:
   - For **Accuracy of Response**:
     ```bash
     npx promptfoo@latest eval --config configs/accuracy_eval.yaml
     ```
   - For **Bias Detection**:
     ```bash
     npx promptfoo@latest eval --config configs/bias_detection.yaml
     ```

3. **View Results:** After evaluation, view or share the results:
   ```bash
   npx promptfoo@latest view
   npx promptfoo@latest share
   ```

## More Details
Refer to `afnan_promptfoo.ipynb` for more details/

