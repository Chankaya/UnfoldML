console.log(tf.getBackend());

//set backend - webGL 100x faster than vanilla CPU[node uses hardware accelerations]
//caveat -- needs explicit memory management
//caveat -- precision on mobile devices
tf.setBackend('cpu');

//avoid blocking UI thread 
tf.matMul(a, b)
//above returns an handle 
x.data();
x.array();



//**making runs faster - shader compilation*/
const model = await tf.loadLayersModel(modelUrl);

// Warmup the model before using real data.
const warmupResult = model.predict(tf.zeros(inputShape));
warmupResult.dataSync();
warmupResult.dispose();

// The second predict() will be much faster
const result = model.predict(userData);

//some flags 
tf.enableProdMode()
tf.enableDebugMode()
