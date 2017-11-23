import React, { Component } from 'react';

import './App.css';

import generateColorSet from './data';
import ColorAccessibilityModel from './neuralNetwork';

const ITERATIONS = 1000;
const TRAINING_SET_SIZE = 5000;
const TEST_SET_SIZE = 10;

class App extends Component {

  testSet;
  trainingSet;
  colorAccessibilityModel;

  constructor() {
    super();

    this.testSet = generateColorSet(TEST_SET_SIZE);
    this.trainingSet = generateColorSet(TRAINING_SET_SIZE);

    this.colorAccessibilityModel = new ColorAccessibilityModel();
    this.colorAccessibilityModel.setupSession(this.trainingSet);

    this.state = {
      currentIteration: 0,
    };
  }

  componentDidMount () {
    requestAnimationFrame(this.tick);
  };

  tick = () => {
    this.setState((state) => ({ currentIteration: state.currentIteration + 1 }));

    if (this.state.currentIteration < ITERATIONS) {
      requestAnimationFrame(this.tick);

      this.colorAccessibilityModel.train(this.state.currentIteration, false);
    }
  };

  render() {
    const { currentIteration } = this.state;

    return (
      <div className="app">
        <div>
          <h1>Neural Network for Font Color Accessibility</h1>
          <p>Iterations: {currentIteration}</p>
        </div>

        <div className="content">
          <div className="content-item">
            <ActualTable
              testSet={this.testSet}
            />
          </div>

          <div className="content-item">
            <InferenceTable
              model={this.colorAccessibilityModel}
              testSet={this.testSet}
            />
          </div>
        </div>
      </div>
    );
  }
}

const ActualTable = ({ testSet }) =>
  <div>
    <p>Programmatically Computed</p>

    {Array(TEST_SET_SIZE).fill(0).map((v, i) =>
      <ColorBox
        key={i}
        rgbInput={testSet.rawInputs[i]}
        rgbTarget={testSet.rawTargets[i]}
      />
    )}
  </div>

const InferenceTable = ({ testSet, model }) =>
  <div>
    <p>Neural Network Computed</p>
    {Array(TEST_SET_SIZE).fill(0).map((v, i) =>
      <ColorBox
        key={i}
        rgbInput={testSet.rawInputs[i]}
        rgbTarget={model.predict(testSet.rawInputs[i])}
      />
    )}
  </div>

const ColorBox = ({ rgbInput, rgbTarget }) =>
  <div className="color-box" style={{ backgroundColor: getRgbStyle(rgbInput) }}>
    <span style={{ color: getRgbStyle(rgbTarget) }}>
      <RgbString rgb={rgbInput} />
    </span>
  </div>

const RgbString = ({ rgb }) =>
  `rgb(${rgb.toString()})`

const getRgbStyle = (rgb) =>
  `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`

export default App;
