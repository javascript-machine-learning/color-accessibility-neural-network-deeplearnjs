import {
  Array1D,
  InCPUMemoryShuffledInputProviderBuilder,
  Graph,
  Session,
  SGDOptimizer,
  NDArrayMathGPU,
  CostReduction,
} from 'deeplearn';

// Encapsulates math operations on the CPU and GPU.
const math = new NDArrayMathGPU();

class ColorAccessibilityModel {
  // Runs training.
  session;

  // An optimizer with a certain initial learning rate. Used for training.
  initialLearningRate = 0.042;
  optimizer;

  // Each training batch will be on this many examples.
  batchSize = 300;

  inputTensor;
  targetTensor;
  costTensor;
  predictionTensor;

  // Maps tensors to InputProviders.
  feedEntries;

  constructor() {
    this.optimizer = new SGDOptimizer(this.initialLearningRate);
  }

  setupSession(trainingSet) {
    const graph = new Graph();

    this.inputTensor = graph.placeholder('input RGB value', [3]);
    this.targetTensor = graph.placeholder('output RGB value', [3]);

    let fullyConnectedLayer = this.createFullyConnectedLayer(graph, this.inputTensor, 0, 64);
    fullyConnectedLayer = this.createFullyConnectedLayer(graph, fullyConnectedLayer, 1, 32);
    fullyConnectedLayer = this.createFullyConnectedLayer(graph, fullyConnectedLayer, 2, 16);

    this.predictionTensor = this.createFullyConnectedLayer(graph, fullyConnectedLayer, 3, 3);
    this.costTensor = graph.meanSquaredCost(this.targetTensor, this.predictionTensor);

    this.session = new Session(graph, math);

    this.generateTrainingData(trainingSet);
  }

  generateTrainingData(trainingSet) {
    math.scope(() => {
      const { rawInputs, rawTargets } = trainingSet;

      const inputArray = rawInputs.map(v => Array1D.new(this.normalizeColor(v)));
      const targetArray = rawTargets.map(v => Array1D.new(this.normalizeColor(v)));

      const shuffledInputProviderBuilder = new InCPUMemoryShuffledInputProviderBuilder([ inputArray, targetArray ]);
      const [ inputProvider, targetProvider ] = shuffledInputProviderBuilder.getInputProviders();

      // Maps tensors to InputProviders.
      this.feedEntries = [
        { tensor: this.inputTensor, data: inputProvider },
        { tensor: this.targetTensor, data: targetProvider },
      ];
    });
  }

  train(step, shouldFetchCost) {
    // Every 42 steps, lower the learning rate by 15%.
    let learningRate = this.initialLearningRate * Math.pow(0.85, Math.floor(step / 42));
    this.optimizer.setLearningRate(learningRate);

    // Train one batch.
    let costValue = -1;
    math.scope(() => {
      const cost = this.session.train(
          this.costTensor,
          this.feedEntries,
          this.batchSize,
          this.optimizer,
          shouldFetchCost ? CostReduction.MEAN : CostReduction.NONE,
        );

      if (!shouldFetchCost) {
        // We only train. We do not compute the cost.
        return;
      }

      // Compute the cost (by calling get), which requires transferring data from the GPU.
      costValue = cost.get();
    });

    return costValue;
  }

  predict(rgb) {
    let accessibleColor = [];

    math.scope((keep, track) => {
      const mapping = [{
        tensor: this.inputTensor,
        data: Array1D.new(this.normalizeColor(rgb)),
      }];

      const values = this.session.eval(this.predictionTensor, mapping).getValues();
      const colors = this.denormalizeColor([ ...values ]);

      // Make sure the values are within range.
      accessibleColor = colors.map(v => Math.round(Math.max(Math.min(v, 255), 0)));
    });

    return accessibleColor;
  }

  // Util

  createFullyConnectedLayer(
    graph,
    inputLayer,
    layerIndex,
    units,
    includeRelu = true,
    includeBias = true
  ) {
    return graph.layers.dense(
      `fully_connected_${layerIndex}`,
      inputLayer,
      units,
      includeRelu ? (x) => graph.relu(x) : undefined,
      includeBias
    );
  }

  normalizeColor(rgb) {
    return rgb.map(v => v / 255);
  }

  denormalizeColor(rgb) {
    return rgb.map(v => v * 255);
  }
}

export default ColorAccessibilityModel;