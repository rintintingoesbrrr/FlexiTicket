# FlexiTicket 
### Neural Network Dynamic Pricing System for Cinema Attendance Optimization

A Multi-Layer Perceptron (MLP) that uses cross-entropy loss for intelligent price calculations in cinema dynamic pricing systems, designed to maximize attendance while optimizing revenue through strategic price reductions.

![MLP working](/images/mlp_visualization.gif)
![Main page](/images/dynamic.png)
![Dynamic pricing](/images/homepage_flexiticket.png)

## Project Overview

FlexiTicket addresses the critical challenge of cinema attendance optimization by implementing a neural network that predicts optimal price reduction strategies. The system analyzes multiple factors including historical attendance, public expectations, and time-based patterns to recommend price adjustments that increase accessibility while maintaining profitability.

### Key Objectives
- **Increase Cinema Accessibility**: Make movies more affordable through intelligent pricing
- **Maximize Attendance**: Fill more seats through strategic price reductions
- **Revenue Optimization**: Balance reduced prices with increased volume
- **Data-Driven Decisions**: Replace guesswork with ML-powered insights

## Model Architecture

### Neural Network Structure
```
Input Layer (5 neurons)
    ↓
Hidden Layer 1 (16 neurons) + ReLU
    ↓
Hidden Layer 2 (16 neurons) + ReLU
    ↓
Output Layer (12 neurons) + Softmax
```

**Total Parameters**: ~500 trainable parameters
**Loss Function**: Cross-Entropy Loss
**Activation**: ReLU (hidden layers), Softmax (output)

## Input Features

The model takes 5 normalized input features:

| Feature | Type | Range | Description |
|---------|------|--------|-------------|
| **Price Reduction Class** | Integer | 0-11 | Historical price reduction category applied |
| **Actual Attendance** | Float | 0-1 | Normalized actual attendance rate |
| **Expected Attendance** | Float | 0-1 | Normalized predicted attendance without intervention |
| **Public Expectations** | Float | 0-1 | Sentiment/expectation score for movie block |
| **Time Block** | Integer | 0-9 | Time slot identifier (e.g., morning, afternoon, evening) |

### Feature Engineering Notes
- **Attendance values** are normalized by theater capacity
- **Public expectations** derived from reviews, ratings, and social media sentiment
- **Time blocks** represent standardized time periods (e.g., 0=early morning, 9=late night)

##  Output Classifications

The model outputs 12 price reduction categories:

| Class | Reduction % | Strategy |
|-------|-------------|----------|
| 0 | No reduction (0%) | Premium pricing |
| 1 | 5% reduction | Minimal discount |
| 2 | 10% reduction | Light discount |
| 3 | 15% reduction | Standard discount |
| 4 | 20% reduction | Moderate discount |
| 5 | 25% reduction | Significant discount |
| 6 | 30% reduction | High discount |
| 7 | 35% reduction | Major discount |
| 8 | 40% reduction | Deep discount |
| 9 | 45% reduction | Maximum discount |
| 10 | 50% reduction | Half-price special |
| 11 | 55%+ reduction | Emergency fill strategy |


### Performance Metrics
- **Primary**: Cross-Entropy Loss (minimization)
- **Secondary**: Classification Accuracy
- **Business**: Attendance Lift % and Revenue Impact

## Expected Outcomes

### Business Impact
- **Attendance Increase**: 15-30% average increase in occupied seats
- **Revenue Optimization**: Maintain 85-95% of full-price revenue through volume
- **Customer Satisfaction**: Increased accessibility and perceived value
- **Market Share**: Competitive advantage through dynamic pricing

### Model Performance Targets
- **Training Accuracy**: >85%
- **Validation Accuracy**: >80%
- **Cross-Entropy Loss**: <0.5
- **Convergence**: Within 50-100 epochs

### Business Logic Validation
- Higher public expectations → Lower price reductions needed
- Large attendance gaps → Higher price reductions recommended
- Peak time blocks → More conservative pricing
- Off-peak periods → More aggressive discounting
