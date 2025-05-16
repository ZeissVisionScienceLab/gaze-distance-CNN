using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ExperimentManager : MonoBehaviour
{

    private ShowTargets showTargets;
    private SaveTracking saveTracking;
    private PlaceTargets placeTargets;
    private SaveMetaData saveMetaData;
    [SerializeField] private GameObject ExperimentObj;
    [SerializeField] private GameObject PlaceTargetsObj;

    [Header("File Names")]
    public string filename = "tracking_data.csv";
    public string targetsFilename = "target_positions.csv";
    public string metadataFilename = "metadata.csv";

    [Header("Data Information (Change every time!)")]
    [SerializeField] private int sceneNumber = 0;
    public int participantID = 0;

    [Header("Experiment Specifications")]
    public bool developerMode = true;
    public bool conditionalTargetPresentation = true;
    [SerializeField] bool calibration = true;
    

    // Start is called before the first frame update
    void Start()
    {
        showTargets = FindObjectOfType<ShowTargets>();
        saveTracking = FindObjectOfType<SaveTracking>();
        placeTargets = FindObjectOfType<PlaceTargets>();
        saveMetaData = FindObjectOfType<SaveMetaData>();

        InitializeParameters();

        if (ExperimentObj.activeSelf)
        {
            
            saveTracking.Initialize();
            Experiment();
        }
        
    }
    void Experiment()
    {
        if (calibration)
        {
            saveTracking.Calibrate();
        }
        showTargets.startExperiment = true;
        

    }
    // Update is called once per frame
    void Update()
    {
        if (ExperimentObj.activeSelf)
        {
            //new calibration, later: every x camera positions
            if (Input.GetKeyDown(KeyCode.Return))
            {
                saveTracking.Msg("Recalibration");
                saveTracking.Calibrate();
            }

        }


    }

    void InitializeParameters() {

        // set up file for target positions
        int index = targetsFilename.LastIndexOf(".csv");
        if (index != -1)
        {
            string part1 = targetsFilename.Substring(0, index);
            string part2 = targetsFilename.Substring(index);
            targetsFilename = part1 + "_" + sceneNumber.ToString() + part2;
            if(ExperimentObj.activeSelf)
                showTargets.filename = targetsFilename;
            if(PlaceTargetsObj.activeSelf)
                placeTargets.filename = targetsFilename;
        }

        // set up file for eye tracking data
        index = filename.LastIndexOf(".csv");
        if (index != -1)
        {
            string part1 = filename.Substring(0, index);
            string part2 = filename.Substring(index);
            filename = part1 + "_" + sceneNumber.ToString() + "_" + participantID.ToString() + part2;
            if (ExperimentObj.activeSelf) 
                saveTracking.trackingFile = filename;
        }

        // set up file for metadata 
        index = metadataFilename.LastIndexOf(".csv");
        if (index != -1)
        {
            string part1 = metadataFilename.Substring(0, index);
            string part2 = metadataFilename.Substring(index);
            metadataFilename = part1 + "_" + sceneNumber.ToString() + "_" + participantID.ToString() + part2;
            if (ExperimentObj.activeSelf)
                saveMetaData.trackingFile = metadataFilename;
        }
    }
}
