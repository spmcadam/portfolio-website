/* simulation-worker.js  */

/* 1.  Load Pyodide */
importScripts(
    "https://cdn.jsdelivr.net/pyodide/v0.27.4/full/pyodide.js"
  );
  
  let pyReady = (async () => {
    // Create the Pyodide instance
    const pyodide = await loadPyodide();
  
    // Only bring in the packages you really use
    await pyodide.loadPackage(["numpy", "pandas", "matplotlib"]);
  
    // Pull in your Python simulation code and execute it
    const code = await fetch("rocket.py").then(r => r.text());
    pyodide.runPython(code);
  
    /* 2.  Tell the main thread we are ready */
    self.postMessage({ type: "ready" });
  
    return pyodide;
  })();
  
  /* 3.  Respond to “run” requests from the UI */
  self.onmessage = async (e) => {
    if (e.data.type !== "run") return;        // ignore anything else
  
    const { v0, burn, coast } = e.data;
    const pyodide = await pyReady;
  
    const imgTag = await pyodide.runPythonAsync(`
  from __main__ import run_simulation
  run_simulation(${v0}, ${burn}, ${coast})
  `);
  
    /* 4.  Send the result back */
    self.postMessage({ type: "result", imgTag });
  };
  
