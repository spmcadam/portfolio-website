/* -----------------------------------------------------------
   trajectories  -  Web-Worker version
----------------------------------------------------------- */

/* 1.  Load Pyodide inside the worker  */
importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.4/full/pyodide.js");

const pyReady = (async () => {
  const py = await loadPyodide();
  await py.loadPackage(["numpy", "pandas", "matplotlib"]);

  /* bring trajectories.py into __main__ so run_simulation is defined */
  const code = await fetch("trajectories.py").then(r => r.text());
  py.runPython(code);

  /* tell the main thread we’re good to go */
  self.postMessage({ type: "ready" });
  return py;
})();

/* 2.  Run a simulation when asked */
self.onmessage = async (e) => {
  if (e.data.type !== "run") return;

  try {
    const { planet, alt, vmin, vmax, nvels } = e.data;
    const py   = await pyReady;

    /* build and run the Python call */
    const imgTag = await py.runPythonAsync(`
from __main__ import run_simulation
run_simulation(${JSON.stringify(planet)},
               ${parseFloat(alt)},
               ${parseFloat(vmin)},
               ${parseFloat(vmax)},
               ${parseInt(nvels, 10)})
`);

    self.postMessage({ type: "result", imgTag });

  } catch (err) {
    self.postMessage({
      type:    "error",
      message: err.message || String(err),
      stack:   err.stack   || null
    });
  }
};

