// simulation-worker.js

// 1) Load Pyodide in the worker
importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.4/full/pyodide.js");

let pyodideReady = loadPyodide().then(async pyodide => {
  // load only the packages you actually need
  await pyodide.loadPackage(["numpy", "matplotlib"]);
  // fetch & run your trajectories.py so run_simulation is defined
  const code = await fetch("trajectories.py").then(r=>r.text());
  pyodide.runPython(code);
  return pyodide;
});

// 2) Listen for messages from the main thread
self.onmessage = async (e) => {
  const { planet, alt, vmin, vmax, nvels } = e.data;
  const pyodide = await pyodideReady;

  // 3) Do the simulation off the main thread
  const imgTag = await pyodide.runPythonAsync(`
from __main__ import run_simulation
run_simulation(${JSON.stringify(planet)},
               ${JSON.stringify(alt)},
               ${JSON.stringify(vmin)},
               ${JSON.stringify(vmax)},
               ${JSON.stringify(nvels)})
`);

  // 4) Send the <img> back to the UI
  self.postMessage({ imgTag });
};
