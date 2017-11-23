import React, { Component } from 'react';

import './App.css';

import generateColorSet from './data';
import ColorAccessibilityModel from './neuralNetwork';

const ITERATIONS = 750;
const TRAINING_SET_SIZE = 1500;
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
      cost: -42,
    };
  }

  componentDidMount () {
    requestAnimationFrame(this.tick);
  };

  tick = () => {
    this.setState((state) => ({ currentIteration: state.currentIteration + 1 }));

    if (this.state.currentIteration < ITERATIONS) {
      requestAnimationFrame(this.tick);

      let computeCost = !(this.state.currentIteration % 5);
      let cost = this.colorAccessibilityModel.train(this.state.currentIteration, computeCost);

      if (cost > 0) {
        this.setState(() => ({ cost }));
      }
    }
  };

  render() {
    const { currentIteration, cost } = this.state;

    return (
      <div className="app">
        <div>
          <h1>Neural Network for Font Color Accessibility</h1>
          <p>Iterations: {currentIteration}</p>
          <p>Cost: {cost}</p>
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
        rgbTarget={fromClassifierToRgb(testSet.rawTargets[i])}
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
        rgbTarget={fromClassifierToRgb(model.predict(testSet.rawInputs[i]))}
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

const fromClassifierToRgb = (classifier) =>
  classifier[0] > classifier[1]
    ? [ 255, 255, 255 ]
    : [ 0, 0, 0 ]

export default App;
