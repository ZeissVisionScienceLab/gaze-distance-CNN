using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using System.IO;
using System;
using TMPro;
using System.Globalization;
using Unity.Mathematics;

public class ReplayManager : MonoBehaviour
{
    public string trackingDataFilename = "tracking_data.csv";
    public int participantID = 0;
    public int sceneID = 0;

    public enum Scene{training, indoor, outdoor};
    public Scene scene = Scene.training;
    [HideInInspector] 

    public string etDataFilepath = Application.dataPath + "/../resources/etdata/";
    public string depthDataFilename = "depth_data.bin";
    private List<Vector3> cameraPositions = new List<Vector3>();
    private List<Quaternion> cameraRotations = new List<Quaternion>();
    private List<Vector3> targetPositions = new List<Vector3>();
    private List<Vector3> leftEyeOrigin = new List<Vector3>();
    private List<Vector3> rightEyeOrigin = new List<Vector3>();
    private List<Vector3> leftEyeGaze = new List<Vector3>();
    private List<Vector3> rightEyeGaze = new List<Vector3>();
    private List<Vector3> combinedEyeOrigin = new List<Vector3>();
    private List<Vector3> combinedEyeGaze = new List<Vector3>();
    private List<Vector3> combinedEyeGazeCamera = new List<Vector3>();

    

    private PostProcessing postProcessing;
    private ExtractDepthData extractDepthData;
    public Camera mainCamera;
    public Camera depthCamera;


    public Camera depthExtractionCamera;

    public int camPosCounter = -1;
    

    void Start()
    {
        // required for depth shader!!!
        depthCamera.depthTextureMode |= DepthTextureMode.Depth;
        depthExtractionCamera.depthTextureMode |= DepthTextureMode.Depth;

        System.Threading.Thread.CurrentThread.CurrentCulture = new CultureInfo("de-DE");
        System.Globalization.CultureInfo customCulture = (System.Globalization.CultureInfo)System.Threading.Thread.CurrentThread.CurrentCulture.Clone();
        customCulture.NumberFormat.NumberDecimalSeparator = ".";
        System.Threading.Thread.CurrentThread.CurrentCulture = customCulture;


        // change data path to Application.dataPath + "/../resources/etdata/indoor", training or outdoor depending on scene
        if (scene == Scene.indoor)
        {
            etDataFilepath = Application.dataPath + "/../resources/etdata/indoor/";
        }

        else if (scene == Scene.outdoor)
        {
            etDataFilepath = Application.dataPath + "/../resources/etdata/outdoor/";
        }
        else
        {
            Debug.Log("Training ReplayManager");
            etDataFilepath = Application.dataPath + "/../resources/etdata/training/";
        }
        
        postProcessing = FindObjectOfType<PostProcessing>();
        extractDepthData = FindObjectOfType<ExtractDepthData>();

        Debug.Log("Main Camera pixel Width" + mainCamera.pixelWidth);
        Debug.Log("Depth camera pixel Width" + depthCamera.pixelWidth);
        Debug.Log("Depth Extraction Camera pixel Width" + depthExtractionCamera.pixelWidth);
        Debug.Log("screen width: " + Screen.width);


        // for height
        Debug.Log("Main Camera pixel Height" + mainCamera.pixelHeight);
        Debug.Log("Depth camera pixel Height" + depthCamera.pixelHeight);
        Debug.Log("Depth Extraction Camera pixel Height" + depthExtractionCamera.pixelHeight);
        Debug.Log("screen height: " + Screen.height);
        
        StartCoroutine(iterateOverFiles());
    }

    void Initialize()
    {
        // get particpant id (last number between _ and .)
        var ids = HelperFunctions.ExtractSceneAndParticipantId(trackingDataFilename);

        participantID = ids.participantId;
        sceneID = ids.sceneId;

        depthDataFilename = "depth_data_" + sceneID + "_" + participantID + ".bin";

        extractDepthData.Initialize(depthDataFilename);

    }

    private void readFromCSV(string filename)
    {
        targetPositions.Clear();
        leftEyeOrigin.Clear();
        leftEyeGaze.Clear();
        rightEyeOrigin.Clear();
        rightEyeGaze.Clear();
        combinedEyeOrigin.Clear();
        combinedEyeGaze.Clear();
        combinedEyeGazeCamera.Clear();
        cameraPositions.Clear();
        cameraRotations.Clear();
        


        // read eye tracking data from file, compute necessary properties
        using (var reader = new StreamReader(filename))
        {
            // skip header
            if (!reader.EndOfStream)
            {
                reader.ReadLine();
            }

            while (!reader.EndOfStream)
            {

                var line = reader.ReadLine();
                var values = line.Split(';');
                Vector3 targetPos = new Vector3(float.Parse(values[20]), float.Parse(values[21]), float.Parse(values[22]));
                Vector3 leftEyeO = new Vector3(float.Parse(values[1]), float.Parse(values[2]), float.Parse(values[3]));
                Vector3 leftEyeG = new Vector3(float.Parse(values[4]), float.Parse(values[5]), float.Parse(values[6]));
                Vector3 rightEyeO = new Vector3(float.Parse(values[7]), float.Parse(values[8]), float.Parse(values[9]));
                Vector3 rightEyeG = new Vector3(float.Parse(values[10]), float.Parse(values[11]), float.Parse(values[12]));

                Vector3 combinedEyeG = leftEyeG + rightEyeG;
                Vector3 combinedEyeO = (leftEyeO + rightEyeO) / 2f;

                Vector3 cameraPos = new Vector3(float.Parse(values[27]), float.Parse(values[28]), float.Parse(values[29]));
                Quaternion cameraRot = new Quaternion(float.Parse(values[30]), float.Parse(values[31]), float.Parse(values[32]), float.Parse(values[33]));

                
                Vector3 combinedEyeGazeC = cameraPos + cameraRot * combinedEyeG;
                combinedEyeGazeCamera.Add(combinedEyeGazeC);

                targetPositions.Add(targetPos);
                leftEyeOrigin.Add(leftEyeO);
                leftEyeGaze.Add(leftEyeG);
                rightEyeOrigin.Add(rightEyeO);
                rightEyeGaze.Add(rightEyeG);

                combinedEyeOrigin.Add(combinedEyeO);
                combinedEyeGaze.Add(combinedEyeG);

                cameraPositions.Add(cameraPos);
                cameraRotations.Add(cameraRot);
            }
        }
    }

    private IEnumerator iterateOverFiles(){
        // iterate over all files in directory depthDataFilepath and save name in list
        string[] files = Directory.GetFiles(etDataFilepath, "*.csv");
        Debug.Log("Files: " + files[0]);
        foreach (string file in files)
        {
            Debug.Log("File: " + file);

            //yield return null;
            trackingDataFilename = file;

            // set filenames properly
            Initialize();

            // read data from file
            readFromCSV(file);
            yield return null;
            
            // iterate over camera positions and save depth data
            yield return StartCoroutine(iterateOverCameraPositions());
        }
    }

    private IEnumerator iterateOverCameraPositions(){

        for (int i = 0; i < cameraPositions.Count; i++)
        {
            // set camera position and rotation
            mainCamera.transform.position = cameraPositions[i];
            mainCamera.transform.rotation = cameraRotations[i];

            Vector3 pos = depthExtractionCamera.WorldToViewportPoint(targetPositions[i]);
            Vector3 gaze = depthExtractionCamera.WorldToViewportPoint(combinedEyeGazeCamera[i]);

            float dist = math.sqrt(math.pow(targetPositions[i].x - cameraPositions[i].x, 2) + math.pow(targetPositions[i].y - cameraPositions[i].y, 2));

            postProcessing.mat.SetVector("_Gaze", gaze);
            extractDepthData.mat.SetVector("_Gaze", gaze);
            yield return null;

            // Start and wait for execution of extractdepthdata.CameraToTexture
            yield return StartCoroutine(extractDepthData.CameraToDepthTexture(gaze));
        }
    }


}
