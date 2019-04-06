//layers api -- sequential model
const model = tf.sequential({
    layers: [
      tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
      tf.layers.dense({units: 10, activation: 'softmax'}),
    ]
   });

//using add()
const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

//layers api -- functional model
const input = tf.input({shape: [784]});
const dense1 = tf.layers.dense({units: 32, activation: 'relu'}).apply(input);
const dense2 = tf.layers.dense({units: 10, activation: 'softmax'}).apply(dense1);
const model = tf.model({inputs: input, outputs: dense2});

//save model -- local storage, indexed db, download
const saveResult = await model.save('localstorage://my-model-1');

//load model
const model = await model.load('localstorage://my-model-1');

//custom layers 
class SquaredSumLayer extends tf.layers.Layer {
    constructor() {
      super({});
    }
    // In this case, the output is a scalar.
    computeOutputShape(inputShape) { return []; }
   
    // call() is where we do the computation.
    call(input, kwargs) { return input.square().sum();}
   
    // Every layer needs a unique name.
    getClassName() { return 'SquaredSum'; }
   }

//test custom layer
const t = tf.tensor([-2, 1, 0, 5]);
const o = new SquaredSumLayer().apply(t);
o.print();

//core api -- no serialisation, more
const w1 = tf.variable(tf.randomNormal([784, 32]));
const b1 = tf.variable(tf.randomNormal([32]));
const w2 = tf.variable(tf.randomNormal([32, 10]));
const b2 = tf.variable(tf.randomNormal([10]));

function model(x) {
  return x.matMul(w1).add(b1).relu().matMul(w2).add(b2).softmax();
}




