using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using System.Threading;
using System.Text;
using ViveSR.anipal.Eye;
using Varjo.XR;
using System.Globalization;

public class SaveTracking : MonoBehaviour
{
    public enum ETProvider
    {
        HTCViveSRanipal,
        Varjo,
        PupilNeon,
        MetaQuest
    }
    public ETProvider etprovider = ETProvider.HTCViveSRanipal;
    //public GameObject trackedGameobject;

    //public bool savePosition = true;
    //public bool saveOrientation = true;
    //public bool saveTargetData = true;
    Queue trackingDataQueue = new Queue();
    static string msgBuffer = "";

    //private float eccentricity = 0; // target eccentricity values
    //private float meridian = 0; // target meridian values
    //private float distance = 0;

  

    // Enum to define different options
    public enum TrackingOptions
    {
        localTransform,
        globalTransform,
    }

    // Define a class to hold the dropdown option and associated GameObject
    [Serializable]
    public class TrackedObjectOptions
    {
        public TrackingOptions trackingOptions;
        public GameObject gameObject;
    }
    
    [Header("Object Tracking")]
    // List to hold the variables with dropdown options and associated GameObjects
    [SerializeField]
    private List<TrackedObjectOptions> trackedObjectList = new List<TrackedObjectOptions>();

    //public ValidationManager valMan;
    [HideInInspector] public string trackingFile;

    private Vector3 combinedEyeOrigin;
    private Vector3 combinedEyeGaze;
    private float gazeDistance; // gaze distance as returned by eye tracking provider
    private bool isTracking = false;

    [HideInInspector] public Vector3 leftEyeOrigin;
    [HideInInspector] public Vector3 rightEyeOrigin;
    [HideInInspector] public Vector3 leftEyeGaze;
    [HideInInspector] public Vector3 rightEyeGaze;


    private ExperimentManager experimentManager;

    void Start()
    {
        // make sure the framework status is WORKING
        Debug.Log(SRanipal_Eye_Framework.Status);
        Debug.Log(SRanipal_Eye_Framework.FrameworkStatus.WORKING);
        System.Threading.Thread.CurrentThread.CurrentCulture = new CultureInfo("de-DE");


    }
    void Update()
    {
        if(isTracking)
        {
            QueueTrackingData();
        }
    }

    public void Initialize()
    {
        experimentManager = FindObjectOfType<ExperimentManager>();
        trackingFile = experimentManager.filename;
         //TODO add relative path to output folder maybe?
        // TODO: check if csv file already exists and change filename if necessary (e.g. _01...)
        WriteHeader();
        InvokeRepeating("Save", 0.0f, 1.0f); // save data to file every second
    }

    public void Calibrate()
    {
        Debug.Log("Starting eye tracking calibration");
        switch (etprovider)
        {
            case ETProvider.HTCViveSRanipal:
                SRanipal_Eye.LaunchEyeCalibration();
                break;
            case ETProvider.Varjo:
                VarjoEyeTracking.RequestGazeCalibration();
                break;
            default:
                break;
        }
    }
    // start with queueing tracking data
    public void StartTracking()
    {
        isTracking = true;
    }
    
    // stop tracking
    public void StopTracking()
    {
        isTracking = false;
        WriteTrackingData(); // perform additional file writing to empty the queue
    } 

    public void Msg(string msg)
    {
        msgBuffer = msg;    
    }

    public void GetGazeData()
    {
        switch (etprovider)
        {
            case ETProvider.HTCViveSRanipal:
                VerboseData eyeData;
                SRanipal_Eye.GetVerboseData(out eyeData);
                leftEyeOrigin = 0.001f * eyeData.left.gaze_origin_mm; // convert mm to m
                leftEyeOrigin.x *= -1; // invert x for gaze to fit the left handed Unity coordinate system
                rightEyeOrigin = 0.001f * eyeData.right.gaze_origin_mm;
                rightEyeOrigin.x *= -1;
                leftEyeGaze = eyeData.left.gaze_direction_normalized;
                leftEyeGaze.x *= -1;
                rightEyeGaze = eyeData.right.gaze_direction_normalized;
                rightEyeGaze.x *= -1;

                combinedEyeOrigin = 0.001f * eyeData.combined.eye_data.gaze_origin_mm;
                combinedEyeOrigin.x *= -1;
                combinedEyeGaze = eyeData.combined.eye_data.gaze_direction_normalized;
                combinedEyeGaze.x *= -1;
                gazeDistance = eyeData.combined.convergence_distance_mm * 0.001f;
                break;
            case ETProvider.Varjo:
                VarjoEyeTracking.GazeData gazeData = VarjoEyeTracking.GetGaze();
                leftEyeOrigin = gazeData.left.origin;
                rightEyeOrigin = gazeData.right.origin;
                leftEyeGaze = gazeData.left.forward;
                rightEyeGaze = gazeData.right.forward;
                combinedEyeOrigin = gazeData.gaze.origin;
                combinedEyeGaze = gazeData.gaze.forward;
                gazeDistance = gazeData.focusDistance;
                break;
            default:
                leftEyeOrigin = -Vector3.forward;
                rightEyeOrigin = -Vector3.forward;
                leftEyeGaze = -Vector3.forward;
                rightEyeGaze = -Vector3.forward;
                break;
        }
    }

    // TODO Use string builder or string.join
    // should be much more effiction: https://stackoverflow.com/questions/21078/most-efficient-way-to-concatenate-strings
    void QueueTrackingData()
    {   
        StringBuilder datasetLine = new StringBuilder(600); // adjust capacity to your needs
        // timestamp: use time at beginning of frame. What makes sense here? Use eye tracker timestamp?
        datasetLine.Append(Time.time.ToString("F10") +";"); 

        GetGazeData();
        // eyetracking data
        // write left eye origin
        datasetLine.Append(leftEyeOrigin.x.ToString("F10") + ";" + leftEyeOrigin.y.ToString("F10") + ";" + leftEyeOrigin.z.ToString("F10") + ";");
        datasetLine.Append(leftEyeGaze.x.ToString("F10") + ";" + leftEyeGaze.y.ToString("F10") + ";" + leftEyeGaze.z.ToString("F10") + ";");
        datasetLine.Append(rightEyeOrigin.x.ToString("F10") + ";" + rightEyeOrigin.y.ToString("F10") + ";" + rightEyeOrigin.z.ToString("F10") + ";");
        datasetLine.Append(rightEyeGaze.x.ToString("F10") + ";" + rightEyeGaze.y.ToString("F10") + ";" + rightEyeGaze.z.ToString("F10") + ";");
        datasetLine.Append(combinedEyeOrigin.x.ToString("F10") + ";" + combinedEyeOrigin.y.ToString("F10") + ";" + combinedEyeOrigin.z.ToString("F10") + ";");
        datasetLine.Append(combinedEyeGaze.x.ToString("F10") + ";" + combinedEyeGaze.y.ToString("F10") + ";" + combinedEyeGaze.z.ToString("F10") + ";");
        datasetLine.Append(gazeDistance.ToString("F10") + ";");



        foreach (TrackedObjectOptions trackedObject in trackedObjectList)
        {
            switch (trackedObject.trackingOptions)
            {
                case TrackingOptions.localTransform:
                    datasetLine.Append(trackedObject.gameObject.transform.localPosition.x.ToString("F10") + ";");
                    datasetLine.Append(trackedObject.gameObject.transform.localPosition.y.ToString("F10") + ";");
                    datasetLine.Append(trackedObject.gameObject.transform.localPosition.z.ToString("F10") + ";");
                    datasetLine.Append(trackedObject.gameObject.transform.localRotation.x.ToString("F10") + ";");
                    datasetLine.Append(trackedObject.gameObject.transform.localRotation.y.ToString("F10") + ";");
                    datasetLine.Append(trackedObject.gameObject.transform.localRotation.z.ToString("F10") + ";");
                    datasetLine.Append(trackedObject.gameObject.transform.localRotation.w.ToString("F10") + ";");
                    break;
                case TrackingOptions.globalTransform:
                    datasetLine.Append(trackedObject.gameObject.transform.position.x.ToString("F10") + ";");
                    datasetLine.Append(trackedObject.gameObject.transform.position.y.ToString("F10") + ";");
                    datasetLine.Append(trackedObject.gameObject.transform.position.z.ToString("F10") + ";");
                    datasetLine.Append(trackedObject.gameObject.transform.rotation.x.ToString("F10") + ";");
                    datasetLine.Append(trackedObject.gameObject.transform.rotation.y.ToString("F10") + ";");
                    datasetLine.Append(trackedObject.gameObject.transform.rotation.z.ToString("F10") + ";");
                    datasetLine.Append(trackedObject.gameObject.transform.rotation.w.ToString("F10") + ";");
                    break;
                default:
                    Debug.LogError("Unknown option selected for " + trackedObject.gameObject.name);
                    break;
            }
        }

        // buffered message
        if (!String.IsNullOrEmpty(msgBuffer))
        {
            datasetLine.Append(msgBuffer + ";");
            msgBuffer = "";
        }
        trackingDataQueue.Enqueue(datasetLine.ToString());
    }

    void Save()
    {
        var thread = new Thread(WriteTrackingData);
        thread.Start();
    }

    void WriteTrackingData()
    {
        FileStream stream = new FileStream(trackingFile, FileMode.Append, FileAccess.Write);
        using (StreamWriter sw = new StreamWriter(stream)) //true for append
        {
            string datasetLine;
            // dequeue trackingDataQueue until empty
            while (trackingDataQueue.Count > 0)
            {
                datasetLine = trackingDataQueue.Dequeue().ToString();
                sw.WriteLine(datasetLine); // write to file
            }
            sw.Close(); // close file
            //Debug.Log("End Writing Data");
        }
    }
    void WriteHeader()
    {
        FileStream stream = new FileStream(trackingFile, FileMode.OpenOrCreate);
        using(StreamWriter sw = new StreamWriter(stream))
        {
            string header = "Timestamp;";
            header += "left_eye_origin.x;left_eye_origin.y;left_eye_origin.z;";
            header += "left_eye_gaze.x;left_eye_gaze.y;left_eye_gaze.z;";
            header += "right_eye_origin.x;right_eye_origin.y;right_eye_origin.z;";
            header += "right_eye_gaze.x;right_eye_gaze.y;right_eye_gaze.z;";
            header += "combined_eye_origin.x;combined_eye_origin.y;combined_eye_origin.z;";
            header += "combined_eye_gaze.x;combined_eye_gaze.y;combined_eye_gaze.z;";
            header += "gaze_distance;";

            foreach (TrackedObjectOptions trackedObject in trackedObjectList)
            {
                switch (trackedObject.trackingOptions)
                {
                    case TrackingOptions.localTransform:
                        header += trackedObject.gameObject.name + "_localPosition.x;";
                        header += trackedObject.gameObject.name + "_localPosition.y;";
                        header += trackedObject.gameObject.name + "_localPosition.z;";
                        header += trackedObject.gameObject.name + "_localRotation.x;";
                        header += trackedObject.gameObject.name + "_localRotation.y;";
                        header += trackedObject.gameObject.name + "_localRotation.z;";
                        header += trackedObject.gameObject.name + "_localRotation.w;";
                        break;
                    case TrackingOptions.globalTransform:
                        header += trackedObject.gameObject.name + "_position.x;";
                        header += trackedObject.gameObject.name + "_position.y;";
                        header += trackedObject.gameObject.name + "_position.z;";
                        header += trackedObject.gameObject.name + "_rotation.x;";
                        header += trackedObject.gameObject.name + "_rotation.y;";
                        header += trackedObject.gameObject.name + "_rotation.z;";
                        header += trackedObject.gameObject.name + "_rotation.w;";
                        break;
                    default:
                        Debug.LogError("Unknown option selected for " + trackedObject.gameObject.name);
                        break;
                }
            }
            header += "messages;";
            sw.WriteLine(header);
            sw.Close();
        }
        
        
    }
}