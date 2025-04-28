/* --------------------------------------------------
   simulation-worker.js
-------------------------------------------------- */

/* 1. Load Pyodide */
importScripts(
    "https://cdn.jsdelivr.net/pyodide/v0.27.4/full/pyodide.js"
  );
  
  const pyReady = (async () => {
    const pyodide = await loadPyodide();
    await pyodide.loadPackage(["numpy", "pandas", "matplotlib"]);
  
    // bring your Python code into the worker’s __main__
    const code = await fetch("rocket.py").then(r => r.text());
    pyodide.runPython(code);
  
    /* Tell the UI we’re alive */
    self.postMessage({ type: "ready" });
    return pyodide;
  })();
  
  /* 2. Run a simulation */
  self.onmessage = async (e) => {
    if (e.data.type !== "run") return;
  
    try {
      const { v0, burn, coast } = e.data;
      const pyodide = await pyReady;
  
      /* no leading spaces ⇒ no IndentationError */
      const imgTag = await pyodide.runPythonAsync(`
  from __main__ import run_simulation
  run_simulation(${v0}, ${burn}, ${coast})
  `);
  
      self.postMessage({ type: "result", imgTag });
  
    } catch (err) {
      /* bubble the error up to the main thread */
      self.postMessage({
        type: "error",
        message: err.message || String(err),
        stack: err.stack || null
      });
    }
  };
  
