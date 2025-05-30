# MindsAI Signal Filter DLL

This project builds `MindsAI.dll`, a C++ neural signal filter using Eigen and FFTW for real-time signal analysis and integration with Unity/C#.

## Version
v5.14.01

## Calling Card

Before calling any signal processing functions, you must initialize the provided MindsAI license like:

```cpp
InitializeMindsAILicense("YOUR-LICENSE-KEY");
```
- This only needs to be called once prior to filtering, and should not be called continuously to prevent API request overload errors.
- If the key is active, data should continue streaming.
- If data does not continue streaming or is not filtered, you can print license details via:
 ```cpp
 GetMindsAILicenseMessag(); 
 ``` 
- This will show you the expiration date and whether or not the key is valid.

After initializing, you can call the filter like:
```cpp
ApplyMindsAIFilter(
    double* input,      // Flattened 3D array: [segments][timepoints][channels]
    int segments,       // Number of independent data segments (e.g., 1-4 for real-time or higher for batch trials)
    int timepoints,     // Number of time samples per segment (e.g., 500 for 1 sec @ 500Hz)
    int channels,       // Number of input channels (e.g., EEG electrodes)
    double lambda,      // Regularization coefficient (e.g., 1e-5; tune for filtering strength or use ML-calibrated value)
    double* output      // Flattened 3D array: [segments][channels][timepoints] with filtered data
);
```
The `lambda` hyperparameter controls how much the MindsAI Filter modifies the original signal and should be input on a logarithmic scale between `0` and `0.1`. A lower `lambda` value like `1e-9` causes the Filter to make bolder adjustments for more complex transformations that can highlight structure across `channels`, such as for real-time filtering with shorter `segments` or fewer `timepoints`. A higher `lambda` value like `1e-5` works best with more data (such as 60-second `segments`) for still helpful, but more conservative adjustments.

## Acknowledgments
 [Brainflow](https://brainflow.org/)
 

## Licensing

 MINDSAPPLIED CONFIDENTIAL
 
 [2023] - [2025] MindsApplied Incorporated 
 All Rights Reserved. For commercial use, contact [contact@minds-applied.com]. 
 
 NOTICE:  All information contained herein is, and remains
 the property of MindsApplied Incorporated and its suppliers,
 if any.  The intellectual and technical concepts contained
 herein are proprietary to Adobe Systems Incorporated
 and its suppliers and may be covered by U.S. and Foreign Patents,
 patents in process, and are protected by trade secret or copyright law.
 Dissemination of this information or reproduction of this material
 is strictly forbidden unless prior written permission is obtained
 from MindsApplied Incorporated.
 