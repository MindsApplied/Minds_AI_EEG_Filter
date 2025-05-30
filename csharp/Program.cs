using brainflow;
using brainflow.math;
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;

class Program
{
    static void Main(string[] args)
    {

        // ********************************************
        // * Variable Initialization *
        //Requirements: MindsAI DLL & License Key, Brainflow SDK and Board DLL
        int boardId = (int)BoardIds.SYNTHETIC_BOARD; // * Brainflow Board of Choice
        string MindsAILicense = ""; // * License Key provided by MindsApplied 
        int secondsWindow = 10;// * Number of seconds per window
        int segments = 1; //Number of windows being input (real-time uses 1 window per update)
        double lambda = 1e-8;// * Lambda to be updated based on data size 
        //Certain parameters like Sampling Rate and Channel Number inherited from brainflow.
        //Confirm all for specific implementations.
        // ********************************************


        //Initialize Brainflow
        BrainFlowInputParams inputParams = new BrainFlowInputParams();
        BoardShim boardShim = new BoardShim(boardId, inputParams);

        BoardShim.enable_dev_board_logger();
        boardShim.prepare_session();
        boardShim.start_stream();

        string filePath = Path.Combine(Directory.GetCurrentDirectory(), "brainflow_data.csv");
        //boardShim.add_streamer(filePath);

        int samplingRate = BoardShim.get_sampling_rate(boardId);
        int dataPoints = samplingRate * secondsWindow;

        int[] eegChannels = BoardShim.get_eeg_channels(boardId);
        int channels = eegChannels.Length;

        // Initialize the license (this must happen once before filtering)
        bool licenseOK = MindsAIDLLWrapper.InitializeMindsAILicense(MindsAILicense);
        Console.WriteLine("License initialized: " + (licenseOK ? "Success" : "Failure"));
        Console.WriteLine("License status: " + MindsAIDLLWrapper.GetLicenseMessage());

        Console.WriteLine($"Compiling EEG data for every {secondsWindow} seconds. Press any key to stop...");

        while (!Console.KeyAvailable)
        {
            Thread.Sleep(secondsWindow * 1000);
            double[,] data = boardShim.get_current_board_data(dataPoints);

            int timepoints = data.GetLength(1);
            double[,] eegData = new double[channels, timepoints];
            for (int ch = 0; ch < channels; ch++)
            {
                double[] channelData = data.GetRow(eegChannels[ch]);
                for (int t = 0; t < timepoints; t++)
                    eegData[ch, t] = channelData[t];
            }

            if (licenseOK)
            {
                // Filter data
                double[,] filtered = SignalFiltering.ApplyMindsAIFilter(eegData, segments, channels, timepoints, lambda);

                //Console.WriteLine($"Filtered shape: [{filtered.GetLength(0)} x {filtered.GetLength(1)}]");
                for (int ch = 0; ch < filtered.GetLength(0); ch++)
                {
                    Console.WriteLine($"[Filtered] Channel {ch + 1} Last Value: {filtered[ch, filtered.GetLength(1) - 1]:F4}");
                }
            }
            else
            {
                // No filtering; stream raw EEG
                for (int ch = 0; ch < eegData.GetLength(0); ch++)
                {
                    Console.WriteLine($"[Raw] Channel {ch + 1} Last Value: {eegData[ch, eegData.GetLength(1) - 1]:F4}");
                }
            }
        }

        boardShim.stop_stream();
        boardShim.release_session();
    }
}

// DLL Wrapper
public static class MindsAIDLLWrapper
{
    [DllImport("MindsAI.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bool InitializeMindsAILicense(string licenseKey);

    [DllImport("MindsAI.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void ApplyMindsAIFilter(
        double[] input, int segments, int timepoints, int channels, double lambda, double[] output);

    [DllImport("MindsAI.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr GetMindsAILicenseMessage();

    public static string GetLicenseMessage()
    {
        IntPtr ptr = GetMindsAILicenseMessage();
        return Marshal.PtrToStringAnsi(ptr);
    }
}

// Signal Filtering Input Class
public static class SignalFiltering
{
    public static double[,] ApplyMindsAIFilter(double[,] eegData2D, int segments, int channels, int timepoints, double lambda)
    {
        // Flatten to [segments=1][timepoints][channels]
        double[] reshapedInput = new double[segments * timepoints * channels];
        for (int ch = 0; ch < channels; ch++)
        {
            for (int t = 0; t < timepoints; t++)
            {
                reshapedInput[t * channels + ch] = eegData2D[ch, t];
            }
        }

        double[] outputFlattened = new double[1 * channels * timepoints];
        MindsAIDLLWrapper.ApplyMindsAIFilter(reshapedInput, segments, timepoints, channels, lambda, outputFlattened);

        double[,] filteredData2D = new double[channels, timepoints];
        for (int ch = 0; ch < channels; ch++)
        {
            for (int t = 0; t < timepoints; t++)
            {
                filteredData2D[ch, t] = outputFlattened[ch * timepoints + t];
            }
        }

        return filteredData2D;
    }
}
