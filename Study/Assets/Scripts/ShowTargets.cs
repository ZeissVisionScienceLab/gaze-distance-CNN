using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using System.IO;
using System;
using TMPro;

public class ShowTargets : MonoBehaviour
{
    
    public GameObject camObject;
    private Camera cam;
    public GameObject XRRig;
    public GameObject camOffset;
    
    private TextMeshPro textMesh;

    [Header("Target presentation:")]
    public GameObject target;
    public GameObject scaler;
    public GameObject leftTarget;
    public GameObject rightTarget;
    public List<Vector3> target_positions = new List<Vector3>();
    public List<Vector3> target_normals = new List<Vector3>();
    public List<Vector3> camera_positions = new List<Vector3>();
    public List<Vector3> camera_rotations = new List<Vector3>();
    public float presentTargetFor = 1f;
    public float stimulusPresentationTime = 0.25f;
    public float targetSize = 0.25f;
    public int breakPeriod = 35;
    private System.Random rnd = new System.Random();

    float distance = 0;

    [Header("Point feedback:")]
    [SerializeField] private GameObject floatingPoints;

    [Header("Audio files:")]
    [SerializeField] private AudioClip correctSound;
    [SerializeField] private AudioClip correctBonusSound;
    [SerializeField] private AudioClip correctDoubleBonusSound;
    [SerializeField] private AudioClip wrongSound;
    [SerializeField] private AudioClip applauseSound;
    [SerializeField] private AudioClip levelCompleteSound;
    [SerializeField] private float volume = 1f;

    // conditions
    private int condition = 0; // green = 0, yellow = 1
    private bool correctAnswer = false;
    private bool keyHit = false;
    private float endTime = 0f;
    private bool firstPos = true;

    // points
    private int points = 0;


    private SaveTracking saveTracking;
    private GazeEstimation gazeEstimation;
    private SaveMetaData saveMetaData;
    private ExperimentManager experimentManager;

    [Header("Display current score:")]
    public GameObject ScoreWindow;
    private GameObject scoreWinChild;

    private Vector3 old_cam_pos;

    [HideInInspector]
    public bool startExperiment = false;

    public Color defaultColor;

    [HideInInspector] public string filename;

    Gamepad gamepad;
    public int counter = 0;
    private bool isLeft = true;
    private Coroutine coroutine;
    private bool skipCurrentTarget = false;

    // Start is called before the first frame update
    void Start()
    {

        gamepad = Gamepad.current;
        if(gamepad == null)
        {
            Debug.LogError("No Controller found.");
        }
        saveTracking = FindObjectOfType<SaveTracking>();
        gazeEstimation = FindObjectOfType<GazeEstimation>();
        saveMetaData = FindObjectOfType<SaveMetaData>();
        experimentManager = FindObjectOfType<ExperimentManager>();

        ScoreWindow.SetActive(false);
        cam = camObject.GetComponent<Camera>();
        scoreWinChild = ScoreWindow.transform.GetChild(0).gameObject;
        textMesh = scoreWinChild.GetComponent<TextMeshPro>();

        System.Globalization.CultureInfo customCulture = (System.Globalization.CultureInfo)System.Threading.Thread.CurrentThread.CurrentCulture.Clone();
        customCulture.NumberFormat.NumberDecimalSeparator = ".";
        System.Threading.Thread.CurrentThread.CurrentCulture = customCulture;

        filename = experimentManager.targetsFilename;
        Debug.Log("filename =" + filename);

        getVectorListFromCSV();

        
        target.SetActive(false);

        


    }

    void Update()
    {
        if (startExperiment)
        {
            StartCoroutine(PresentTargets());
            startExperiment = false;

        }
        
        // save participant reactions
        if (gamepad.leftTrigger.ReadValue() >= 0.95f)
        {
            endTime = Time.time;
            keyHit = true;
            correctAnswer = (isLeft && condition == 0) || (!isLeft && condition == 1);
        }

        if (gamepad.rightTrigger.ReadValue() >= 0.95f)
        {
            endTime = Time.time;
            keyHit = true;
            correctAnswer = (isLeft && condition == 1) || (!isLeft && condition == 0);
        }

        // option to skip a target (if it cannot be focused for example)
        if (Input.GetKeyUp(KeyCode.R))
        {
            skipCurrentTarget = true;
            saveMetaData.Msg("Skip target");
        }

        // additional options if developer mode is running
        if (experimentManager.developerMode)
        {
            if (gamepad.xButton.ReadValue() >= 0.95)
            {
                saveTracking.Msg("Remove target");
                saveMetaData.Msg("Remove target");
            }
            if (gamepad.yButton.ReadValue() >= 0.95)
            {
                skipCurrentTarget = true;
                saveTracking.Msg("Skip target");
                saveMetaData.Msg("Skip target");
                saveMetaData.Msg("Skip target");
            }
        }
    }

    private void getVectorListFromCSV()
    {
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

                Vector3 vector = new Vector3(float.Parse(values[0]), float.Parse(values[1]), float.Parse(values[2]));
                Vector3 normal = new Vector3(float.Parse(values[3]), float.Parse(values[4]), float.Parse(values[5]));
                Vector3 camera_pos = new Vector3(float.Parse(values[6]), float.Parse(values[7]), float.Parse(values[8]));
                Vector3 camera_rot = new Vector3(float.Parse(values[9]), float.Parse(values[10]), float.Parse(values[11]));

                target_positions.Add(vector);
                target_normals.Add(normal);
                camera_positions.Add(camera_pos);
                camera_rotations.Add(camera_rot);
            }
        }
        Vector3[][] positions = { target_positions.ToArray(), target_normals.ToArray(), camera_positions.ToArray(), camera_rotations.ToArray() };

        shuffleVectorList();

        
    }

    void shuffleVectorList()
    {
        int blockSize = 5;
        Vector3[][] positions = { target_positions.ToArray(), target_normals.ToArray(), camera_positions.ToArray(), camera_rotations.ToArray() };

        if (target_positions.Count % 5 != 0)
            Debug.Log("Array is not divisible by 5!");

        Debug.Log("outer: " + positions.Length);
        Debug.Log("inner: "+ positions[0].Length);
        int nrBlocks = target_positions.Count / blockSize;
        

        Debug.Log("nrBlocks = " + nrBlocks);

        for(int i = nrBlocks - 1; i >= 1; i--)
        {

            // get shuffle parameters
            int j = rnd.Next(0, i+1);

            int baseI = i * blockSize;
            int baseJ = j * blockSize;

            Debug.Log("i = " + i + "j = " + j);

            
            // get subarrays
            Vector3[][] positionsI = new Vector3[4][];
            Vector3[][] positionsJ = new Vector3[4][];

            for (int x = 0; x < 4; x++)
            {
                positionsI[x] = new Vector3[blockSize];
                positionsJ[x] = new Vector3[blockSize];
                for (int y = 0; y < blockSize; y++)
                {                    
                    positionsI[x][y] = positions[x][baseI + y];
                    positionsJ[x][y] = positions[x][baseJ + y];
                }

            }

            // shuffle subarrays
            for (int a = positionsI.Length; a >= 1; a--)
            {
                int b = rnd.Next(0, a + 1);
                for (int x = 0; x < 4; x++)
                {
                    Vector3 posAcc = positionsI[x][a];
                    positionsI[x][a] = positionsI[x][b];
                    positionsI[x][b] = posAcc;
                }

            }

            for (int a = positionsJ.Length; a >= 1; a--)
            {
                int b = rnd.Next(0, a + 1);
                for (int x = 0; x < 4; x++)
                {
                    Vector3 posAcc = positionsJ[x][a];
                    positionsJ[x][a] = positionsJ[x][b];
                    positionsJ[x][b] = posAcc;
                }

            }

            // switch blocks I and J
            for (int x = 0; x < 4; x++)
            {
                for (int y = 0; y < blockSize; y++)
                {
                    positions[x][baseI + y] = positionsJ[x][y];
                    positions[x][baseJ + y] = positionsI[x][y];
                }
                
            }
        }

        int max = target_positions.Count;

        target_positions = new List<Vector3>();
        target_normals = new List<Vector3>();
        camera_positions = new List<Vector3>();
        camera_rotations = new List<Vector3>();

        Debug.Log("Array legth: " + max);
        Debug.Log("other array length: " + positions.Length);
        for (int i = 0; i < max; i++)
        {
            target_positions.Add(positions[0][i]);
            target_normals.Add(positions[1][i]);
            camera_positions.Add(positions[2][i]);
            camera_rotations.Add(positions[3][i]);
        }

    }


    void updateCameraOffset()
    {
        camOffset.transform.localPosition = (-1) * cam.transform.localPosition;// new Vector3(0,0,0);
        Debug.Log("local cam: " + cam.transform.localPosition + "global cam: " + cam.transform.position);
    }

    void updateScoreWinPos()
    {
        ScoreWindow.transform.position = cam.transform.position;
        Vector3 newEuler = cam.transform.eulerAngles;
        Vector3 newRot = new Vector3(newEuler.x, newEuler.y, 0);
        ScoreWindow.transform.eulerAngles = newRot;
    }

    IEnumerator PresentTarget(Vector3 t)
    {
        
        XRRig.transform.position = camera_positions[target_positions.IndexOf(t)];
        XRRig.transform.eulerAngles = camera_rotations[target_positions.IndexOf(t)];
        Vector3 normal = target_normals[target_positions.IndexOf(t)];

        if (counter != 0 && counter % 5 == 0)
        {
            firstPos = true;
            target.SetActive(false);

            // pause until enter is pressed
            if (counter % breakPeriod == 0)
            {
                SoundFXManager.instance.PlaySoundFXClip(applauseSound, target.transform, volume);
                textMesh.text = "Your current score is " + points.ToString() + "\n Have a small break.";
                updateScoreWinPos();
                ScoreWindow.SetActive(true);
                yield return new WaitUntil(() => Input.GetKeyUp(KeyCode.Space));
            }
            else
            {
                SoundFXManager.instance.PlaySoundFXClip(levelCompleteSound, target.transform, volume);
            }
            textMesh.text = "Your current score is " + points.ToString() + "\n Press B to continue";
            updateScoreWinPos();

            ScoreWindow.SetActive(true);
            yield return new WaitUntil(() => (gamepad.bButton.ReadValue() >= 0.95f));// Input.GetKeyUp(KeyCode.Space));
            ScoreWindow.SetActive(false);

            updateCameraOffset();

            yield return new WaitForSeconds(1f);
            target.SetActive(true);

        }
        counter += 1;

        old_cam_pos = camera_positions[target_positions.IndexOf(t)];
        target.SetActive(true);
        
        
        // set target position, rotation and scale

        // get normal vector of object at target position
        Vector3 origin = XRRig.transform.position; 
        Vector3 direction = t - origin;
        Vector3 scaleOld = target.transform.localScale;

        // get distance to object
        distance = Vector3.Distance(cam.transform.position, t);

        // get angle between normal and ray direction
        Vector2 a = new Vector2((-1) * normal.x, (-1) * normal.z);
        Vector2 b = new Vector2(direction.x, direction.z);
        float angle_xz = (float)Vector2.Angle(a, b);

        Vector2 c = new Vector2((-1) * normal.y, (-1) * normal.z);
        Vector2 d = new Vector2(direction.y, direction.z);
        float angle_yz = (float)Vector2.Angle(c, d);

        // scale x and y coordinates of the target such that it looks similar from each direction and distance
        float x_scaled = targetSize * distance; 
        float y_scaled = targetSize * distance; 
        float z_scaled = scaleOld.z;

        //target.transform.localScale = new Vector3(x_scaled, y_scaled, z_scaled);

        Vector3 targetDir = target.transform.position - cam.transform.position;

        //yield return MoveTarget(t, Quaternion.LookRotation(normal, targetDir), new Vector3(x_scaled, y_scaled, z_scaled), 1f);
        yield return MoveTarget(t, Quaternion.LookRotation(normal), new Vector3(x_scaled, y_scaled, z_scaled), 1f);


        // check if participant focuses the target
        gazeEstimation.checkForFocus = true;
        gazeEstimation.targetFocused = false;

        while (!gazeEstimation.targetFocused)
        {
            // option to abort current target presentation
            if (skipCurrentTarget)
            {
                skipCurrentTarget = false;
                yield break;
            }
            
            yield return null;
        }

        // start eye tracking and target presentation countdown
        saveTracking.StartTracking();

        // add 0-2s randomly
        float rndAdd = (float) rnd.NextDouble() * 1;
        // track during fixed presentation time
        yield return new WaitForSeconds(presentTargetFor);
        saveTracking.StopTracking();

        // add random amount of time for less predictable target presentation
        yield return new WaitForSeconds(rndAdd);


        // stimulus presentation

        // randomly generate number 0 or 1
        condition = rnd.Next(0, 2);
        keyHit = false; //only consider keypresses from now on
        if (condition == 0)
            leftTarget.GetComponent<Renderer>().material.color = Color.yellow;
        else
            rightTarget.GetComponent<Renderer>().material.color = Color.yellow;

        // change color back to default when either space is pressed or after stimulusPresentationTime

        // get current time
        float startTime = Time.time;
        while (!keyHit && Time.time < startTime + stimulusPresentationTime)
        {
            yield return null;
        }

        leftTarget.GetComponent<Renderer>().material.color = defaultColor;
        rightTarget.GetComponent<Renderer>().material.color = defaultColor;

        while (!keyHit)
        {
            yield return null;
        }

        float reactionTime = endTime - startTime;

        int addPoints = 0;

        //isLeft = gazeEstimation.isLeft(leftTarget.transform.position, rightTarget.transform.position);
        isLeft = gazeEstimation.isLeft2(normal, Vector3.up, target.transform.right);
        Debug.Log("is left =" + isLeft);

        yield return null;

        if (correctAnswer)
        {
            if (reactionTime < 0.2f)
                addPoints = 20;
            else if (reactionTime > 0.5f)
                addPoints = 1;
            else
            {
                addPoints = (int)(1 + (reactionTime - 0.5f) / (0.2 - 0.5) * 19);
            }

        }
        else
        {
            addPoints = -5;
        }

        points += addPoints;

        // add point gameobject
        GameObject myFloatingPoints = Instantiate(floatingPoints, t, Quaternion.identity);
        myFloatingPoints.layer = 3;
        myFloatingPoints.transform.forward = (t - cam.gameObject.transform.position).normalized;
        

        if (addPoints > 0)
        {
            myFloatingPoints.transform.GetChild(0).GetComponent<TextMesh>().color = Color.green;
            myFloatingPoints.transform.GetChild(0).GetComponent<TextMesh>().text = "+" + addPoints.ToString();
            if (addPoints >= 11)
            {
                SoundFXManager.instance.PlaySoundFXClip(correctDoubleBonusSound, target.transform, volume);
            }
            else if (addPoints > 8) 
            {
                SoundFXManager.instance.PlaySoundFXClip(correctBonusSound, target.transform, volume);
            }
            else
            {
                SoundFXManager.instance.PlaySoundFXClip(correctSound, target.transform, volume);
            }
        }
        else
        {
            myFloatingPoints.transform.GetChild(0).GetComponent<TextMesh>().color = Color.red;
            myFloatingPoints.transform.GetChild(0).GetComponent<TextMesh>().text = "" + addPoints.ToString();
            SoundFXManager.instance.PlaySoundFXClip(wrongSound, target.transform, volume);
        }

        myFloatingPoints.transform.GetChild(0).GetComponent<TextMesh>().fontSize = Math.Min(500, 20 * (int)distance);

        // save metadata
        saveMetaData.QueueData(t, correctAnswer, reactionTime);
        saveMetaData.WriteData();
        Debug.Log("Points: " + points);

        yield return new WaitForSeconds(1f);
    }

    IEnumerator PresentTargets()
    {

        if (target_positions.Count == 0)
        {
            Debug.Log("No target positions found.");
            yield break;
        }

        textMesh.text = "Please wait for the experimentator \n to start the experiment";
        XRRig.transform.position = camera_positions[0];
        XRRig.transform.eulerAngles = camera_rotations[0];

        updateCameraOffset();

        //updateScoreWinPos();
        ScoreWindow.SetActive(true);
        // wait for monitoring person to press space and for participant to press B
        if (experimentManager.developerMode)
        {
            yield return new WaitUntil(() => (gamepad.aButton.ReadValue() >= 0.95));
        }
        else
        {
            yield return new WaitUntil(() => Input.GetKeyUp(KeyCode.Space));
        }
        ScoreWindow.SetActive(false);
        yield return null;
        textMesh.text = "Press B to start";
        ScoreWindow.SetActive(true);
        yield return new WaitUntil(() => (gamepad.bButton.ReadValue() >= 0.95f));
        ScoreWindow.SetActive(false);


        yield return new WaitForSeconds(1f);

        old_cam_pos = camera_positions[0];



        foreach (Vector3 t in target_positions)
        {
            skipCurrentTarget = false;
            yield return StartCoroutine(PresentTarget(t));
        }

        textMesh.text = "Your final score is " + points.ToString() + "\n Well done!";
        updateScoreWinPos();
        ScoreWindow.SetActive(true);

        // wait 1s due to delay of saving in eyetracking script
        yield return new WaitForSeconds(1f);
        saveTracking.StopTracking();
        
    }

    IEnumerator MoveTarget(Vector3 targetPos, Quaternion targetRot, Vector3 targetScale, float motionTime)
    {
        if (firstPos)
        {
            target.transform.position = targetPos;
            target.transform.rotation = targetRot;
            scaler.transform.localScale = targetScale;
            firstPos = false;
        }
        else
        {
            float elapsedTime = 0f;
            Vector3 startPos = target.transform.position;
            Quaternion startRot = target.transform.rotation;
            Vector3 startScale = scaler.transform.localScale;
            AnimationCurve speedCurve = AnimationCurve.EaseInOut(0f, 0f, 1f, 1f);

            while (elapsedTime < motionTime)
            {
                elapsedTime += Time.deltaTime;
                float t = speedCurve.Evaluate(elapsedTime / motionTime);

                target.transform.position = Vector3.Lerp(startPos, targetPos, t);
                target.transform.rotation = Quaternion.Lerp(startRot, targetRot, t);
                scaler.transform.localScale = Vector3.Lerp(startScale, targetScale, t);
                yield return null; // Wait for the next frame
            }
        }

    }

}
