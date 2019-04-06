require('babel-runtime/regenerator/index')
import * as tf from '@tensorflow/tfjs';
import {MnistData} from './data';

var model;

function createModel(){
    createLogEntry('Create model ...');
    model = tf.sequential();
    createLogEntry('Model created');

    createLogEntry('Add layers...')
    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
      }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
      }));
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
      }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
      }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({
        units: 10,
        kernelInitializer: 'VarianceScaling',
        activation: 'softmax'
      }));
    createLogEntry('Layers created');
    createLogEntry('Start compiling ...');
    model.compile({
        optimizer: tf.train.sgd(0.15),
        loss: 'categoricalCrossentropy'
    });
    createLogEntry('Compiled');

}

let data;
async function load() {
    createLogEntry('Loading MNIST data ...');
    data = new MnistData();
    await data.load();
    createLogEntry('Data loaded successfully');
}

async function train(onIteration) {
    createLogEntry('Training model...');
  
    
    const optimizer = 'rmsprop';
  
    model.compile({
      optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  
   
    const batchSize = 500;
  
    const validationSplit = 0.15;
  
   
    const trainEpochs = 1;
  
    let trainBatchCount = 0;
  
    const trainData = data.getTrainData();
    const testData = data.getTestData();
  
    const totalNumBatches =
        Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) *
        trainEpochs;

    let valAcc;
    await model.fit(trainData.xs, trainData.labels, {
      batchSize,
      validationSplit,
      epochs: trainEpochs,
      callbacks: {
        onBatchEnd: async (batch, logs) => {
          trainBatchCount++;
    
          if (onIteration && batch % 10 === 0) {
            onIteration('onBatchEnd', batch, logs);
          }
          await tf.nextFrame();
        },
        onEpochEnd: async (epoch, logs) => {
          valAcc = logs.val_acc;
          if (onIteration) {
            onIteration('onEpochEnd', epoch, logs);
          }
          await tf.nextFrame();
        }
      }
    });
  const testResult = model.evaluate(testData.xs, testData.labels);
  const testAccPercent = testResult[1].dataSync()[0] * 100;
  const finalValAccPercent = valAcc * 100;
}

async function showPredictions(model) {
    const testExamples = 100;
    const examples = data.getTestData(testExamples);
  

    tf.tidy(() => {
      const output = model.predict(examples.xs);
  
    
      const axis = 1;
      const labels = Array.from(examples.labels.argMax(axis).dataSync());
      const predictions = Array.from(output.argMax(axis).dataSync());
  
      createLogEntry(predictions)
    });
  }

function createLogEntry(entry) {
    document.getElementById('log').innerHTML += '<br>' + entry;
}

async function predict(batch) {
    tf.tidy(() => {
        const input_value = Array.from(batch.labels.argMax(1).dataSync());
        
        const div = document.createElement('div');
        div.className = 'prediction-div';
        const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]));
        const prediction_value = Array.from(output.argMax(1).dataSync());
        const image = batch.xs.slice([0,0], [1, batch.xs.shape[1]]);
        
        const canvas = document.createElement('canvas');
        canvas.className = 'prediction-canvas';
        draw(image.flatten(), canvas);
        const label = document.createElement('div');
        label.innerHTML = 'Original Value: ' + input_value;
        label.innerHTML += '<br>Prediction Value: ' + prediction_value;
        console.log(prediction_value + '-' + input_value);
        if (prediction_value - input_value == 0) {
            label.innerHTML += '<br>Value recognized successfully!';
        } else {
            label.innerHTML += '<br>Recognition failed!';
        }
            
        div.appendChild(canvas);
        div.appendChild(label);
        document.getElementById('predictionResult').appendChild(div);
    });
}

function draw(image, canvas) {
    const [width, height] = [28, 28];
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = image.dataSync();
    for (let i = 0; i < height * width; ++i) {
      const j = i * 4;
      imageData.data[j + 0] = data[i] * 255;
      imageData.data[j + 1] = data[i] * 255;
      imageData.data[j + 2] = data[i] * 255;
      imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

document.getElementById('selectTestDataButton').addEventListener('click', async (el, ev) => {
    const batch = data.getTestData();
    await predict(batch);
});

async function main() {
    createModel();
    await load();
    await train(() => showPredictions(model));
    document.getElementById('selectTestDataButton').disabled = false;
    document.getElementById('selectTestDataButton').innerText = "Randomly Select Test Data And Predict";
}
main();
