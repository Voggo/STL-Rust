# NOTES from Breach implementation

## Setting up Breach

I did the following:

- Install Matlab R2025b and Simulink.
- Clone the Breach repository from GitHub:

  ```bash
  git clone https://github.com/decyphir/breach.git
  ```

- Try to compile the code by running (in root):

   ```matlab
   InstallBreach
   ```

- Had some issues, likely related to the fact I'm on an M4 Pro Mac. I had to:
  - Accept Xcode license: `sudo xcodebuild -license accept`. Source: [MATLAB Central Answer](https://se.mathworks.com/matlabcentral/answers/2180302-mex-failing-to-compile-function)
  - In '@STL_Formula/private/src/robusthom/CompileRobusthom.m', I had to add a flag to the mex command to specify the architecture:

    ```matlab
    LDFLAGS=$LDFLAGS -ld_classic
    ```

    Source: [MATLAB Central Answer](https://se.mathworks.com/matlabcentral/answers/2180302-mex-failing-to-compile-function)
  - I had to do the same in 'Online/m_src/compile_stl_mex.m'.
  - Had to install Java Runtime [Java Runtime for Apple Silicon](https://se.mathworks.com/support/requirements/apple-silicon.html)

After these steps, I was able to compile Breach successfully.

After that, run:

```matlab
InitBreach;
```

I am still not able to run any of the examples, getting errors like:

```matlab
>> BrDemo.AFC_1_Interface
Error using BreachSimulinkSystem/CreateInterface (line 233)
AbstractFuelControl does not exists or is not a valid Simulink filename

Error in BreachSimulinkSystem (line 103)
                    this.CreateInterface(mdl_name);
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Error in BrDemo.AFC_1_Interface (line 37)
BrAFC = BreachSimulinkSystem(mdl);
```

But I was able to write scripts that create signals and evaluate STL formulas on them, which is essentially all we need for now.

## Online Monitoring Approaches in Breach

Breach actually does support RoSI monitoring. It's the approach they call 'online' monitoring in their STL_Eval function.

They also do ‘classic’ monitoring, which is similar to our strict approach. The difference is that they extrapolate into the unknown future. That is, instead of delaying a verdict until it is final, they assume (for now) that the current signal will hold for the remaining of the interval. Only then all data is available, they finalize the verdict.

They provide an optimization step for classic called 'thom' in the STl_Eval function. This removes redundant values. So it only outputs in critical points, i.e. in minima/maxima of the robustness signal or when the robustness crosses zero. This will cause the output to be sparse.
