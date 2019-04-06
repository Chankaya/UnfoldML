// Array to Tensor 
const a = tf.tensor([[4, 5], [3, 4]]);
console.log('shape:', a.shape);

//Reshape 1-d array to tensor 
const shape = [2, 2];
const b = tf.tensor([1, 2, 3, 4], shape);
console.log('shape:', b.shape);

//Possible types -- int32, bool, complex64, string
const type = 'bool';
const a1 = tf.tensor([[1, 2], [3, 4]], shape, type);
console.log('shape:', a1.shape);
console.log('dtype', a1.dtype);

//reshape - reshape size product should match the predessor size
const c = a.reshape([4,1]);
c.print();

//array and data functions of tensor
a.array().then(array => console.log(array));
a.data().then(data => console.log(data));

//synchronous versions -- performance issuses
console.log(a.arraySync());
console.log(a.dataSync());

// some operations -- x^2+y
const x = tf.tensor([1, 2, 3, 4]);
const y = tf.tensor([10, 20, 30, 40]);
const sq = x.square();
const fn = sq.add(y);
fn.print();

//webGL memory management -- explicit memory management needed 
a.dispose()

//tf.tidy
const a = tf.tensor([[1, 2], [3, 4]]);
const negLog = tf.tidy(() => {
  const result = a.square().log().neg();
  return result;
});

//memory tracking 
console.log(tf.memory());