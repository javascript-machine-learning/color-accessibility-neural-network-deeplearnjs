import {
  Array1D,
  InCPUMemoryShuffledInputProviderBuilder,
  Graph,
  Session,
  SGDOptimizer,
  ENV,
  NDArrayMath,
  CostReduction,
} from 'deeplearn';

class ColorAccessibilityModel {
  // Encapsulates math operations on the CPU and GPU.
  math = ENV.math;

  // Runs training.
  session;

  // An optimizer with a certain initial learning rate. Used for training.
  initialLearningRate = 0.06;
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
    this.targetTensor = graph.placeholder('output RGB value', [2]);

    let fullyConnectedLayer = this.createFullyConnectedLayer(graph, this.inputTensor, 0, 64);
    fullyConnectedLayer = this.createFullyConnectedLayer(graph, fullyConnectedLayer, 1, 32);
    fullyConnectedLayer = this.createFullyConnectedLayer(graph, fullyConnectedLayer, 2, 16);

    this.predictionTensor = this.createFullyConnectedLayer(graph, fullyConnectedLayer, 3, 2);
    this.costTensor = graph.meanSquaredCost(this.targetTensor, this.predictionTensor);

    this.session = new Session(graph, this.math);

    this.prepareTrainingSet(trainingSet);
  }

  prepareTrainingSet(trainingSet) {
    const oldMath = ENV.math;
    const safeMode = false;
    const math = new NDArrayMath('cpu', safeMode);
    ENV.setMath(math);

    const { rawInputs, rawTargets } = trainingSet;

    const inputArray = rawInputs.map(v => Array1D.new(this.normalizeColor(v)));
    const targetArray = rawTargets.map(v => Array1D.new(v));

    const shuffledInputProviderBuilder = new InCPUMemoryShuffledInputProviderBuilder([ inputArray, targetArray ]);
    const [ inputProvider, targetProvider ] = shuffledInputProviderBuilder.getInputProviders();

    // Maps tensors to InputProviders.
    this.feedEntries = [
      { tensor: this.inputTensor, data: inputProvider },
      { tensor: this.targetTensor, data: targetProvider },
    ];

    ENV.setMath(oldMath);
  }

  train(step, computeCost) {
    // Every 50 steps, lower the learning rate by 10%.
    let learningRate = this.initialLearningRate * Math.pow(0.90, Math.floor(step / 50));
    this.optimizer.setLearningRate(learningRate);

    // Train one batch.
    let costValue;
    this.math.scope(() => {
      const cost = this.session.train(
        this.costTensor,
        this.feedEntries,
        this.batchSize,
        this.optimizer,
        computeCost ? CostReduction.MEAN : CostReduction.NONE,
      );

      // Compute the cost (by calling get), which requires transferring data from the GPU.
      if (computeCost) {
        costValue = cost.get();
      }
    });

    return costValue;
  }

  predict(rgb) {
    let classifier = [];

    this.math.scope(() => {
      const mapping = [{
        tensor: this.inputTensor,
        data: Array1D.new(this.normalizeColor(rgb)),
      }];

      classifier = this.session.eval(this.predictionTensor, mapping).getValues();
    });

    return [ ...classifier ];
  }

  createFullyConnectedLayer(
    graph,
    inputLayer,
    layerIndex,
    units,
    activationFunction
  ) {
    return graph.layers.dense(
      `fully_connected_${layerIndex}`,
      inputLayer,
      units,
      activationFunction
        ? activationFunction
        : (x) => graph.relu(x)
    );
  }

  normalizeColor(rgb) {
    return rgb.map(v => v / 255);
  }
}

export default ColorAccessibilityModel;