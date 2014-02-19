var camera, scene, renderer;
var geometry, material, mesh;

var rbtData = {};
var on_idx = 0;

init();
animate();

function init() {

  camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, .1, 10000 );
  camera.position.z = 1000;
  var controls = new THREE.OrbitControls (camera);
  controls.addEventListener('change', render);

  scene = new THREE.Scene();

  geometry = new THREE.BoxGeometry( 105, 30, 68 );
  material = new THREE.MeshLambertMaterial( { color: 0xff0000} );

  mesh = new THREE.Mesh( geometry, material );
  scene.add( mesh );

  make_floor();

  hemiLight = new THREE.HemisphereLight( 0xffffff, 0xffffff, 0.6 );
  hemiLight.color.setHSL( 0.6, 1, 0.6 );
  hemiLight.groundColor.setHSL( 0.095, 1, 0.75 );
  hemiLight.position.set( 0, 500, 0 );
  scene.add( hemiLight );

  var dir_light = new THREE.DirectionalLight(0xffffff, .4);
  dir_light.position.set(0, 1, 1);
  scene.add(dir_light);

  var dir_light = new THREE.DirectionalLight(0xffffff, .4);
  dir_light.position.set(1, 1, 1);
  scene.add(dir_light);

  renderer = new THREE.WebGLRenderer();
  renderer.setSize( window.innerWidth, window.innerHeight );

  clock = new THREE.Clock();

  container = document.getElementById('container')
  stats = new Stats();
  stats.domElement.style.position = 'absolute';
  stats.domElement.style.top = '0px';
  stats.domElement.style.zIndex = 100;
  container.appendChild( stats.domElement );

  //Load up json file
  console.log("start loading");
  var dat = $.getJSON('test.json', function(data) {
    console.log("done loading");
    rbtData = data;
    clock.start();
  });

  //Add renderer
  container.appendChild( renderer.domElement );
}

function make_floor() {
  var geometry = new THREE.BoxGeometry( 5000, 20, 5000);
  var material = new THREE.MeshLambertMaterial( { color: 0x333333} );
  var mesh = new THREE.Mesh( geometry, material );
  mesh.position.y = -10;
  scene.add( mesh );
}

function set_state(idx) {
  var tx = rbtData.ukf.x[on_idx];
  var ty = rbtData.ukf.y[on_idx];
  var tz = rbtData.ukf.z[on_idx];
  mesh.position.x = tx;
  //Sswap z and y
  mesh.position.y = tz + 15; //Make up for box height of 30
  mesh.position.z = ty;

  var tyaw = rbtData.ukf.yaw[on_idx];
  var tpitch = rbtData.ukf.pitch[on_idx];
  var troll = rbtData.ukf.roll[on_idx];
  var toRad = 3.1415 / 180.0;

  //because of zy swap, these change these aswell
  mesh.rotation.y = tyaw * toRad
  mesh.rotation.z = tpitch * toRad;
  mesh.rotation.x = troll * toRad;
}

function animate() {
  // note: three.js includes requestAnimationFrame shim
  requestAnimationFrame( animate );
  render( scene, camera );
  if (! rbtData.time) {
    return;
  }

  var new_time = clock.getElapsedTime();
  var target_time = new_time / 2.0;
  while (rbtData.time[on_idx] <= target_time && rbtData.time.length > on_idx) {
    on_idx += 1;
  }

  if (rbtData.time.length <= on_idx) {
    clock.stop();
    clock.elapsedTime = 0
    clock.start();
    on_idx = 0;
  }

  set_state(on_idx);
}

function render() {
  renderer.render(scene, camera);
  stats.update();
}
