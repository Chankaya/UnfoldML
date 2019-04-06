//npm install -g parcel-bundler
//npm install --save-dev babel-plugin-transform-runtime babel-runtime
//npm install @tensorflow/tfjs

require('babel-runtime/regenerator/index')
import * as tf from '@tensorflow/tfjs';
import {MnistData} from './mnist-data';

var model;

function createModel(){}

let data;
async function load() {
    createLogEntry('Loading MNIST data ...');
    data = new MnistData();
    await data.load();
    createLogEntry('Data loaded successfully');
}

async function train(){}

async function predict(){}

document.getElementById('selectTestDataButton').addEventListener('click', async (el, ev) => {
    const batch = data.getTestData();
    await predict(batch);
});

async function main() {
    createModel();
    await load();
    await train(() => showPredictions(model));
    //your buttons and your flow
}
main();

