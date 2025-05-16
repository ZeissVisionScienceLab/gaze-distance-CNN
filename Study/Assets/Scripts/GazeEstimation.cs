using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public class GazeEstimation : MonoBehaviour
{
    private SaveTracking saveTracking;
    private Vector3 leftEyeOrigin;
    private Vector3 rightEyeOrigin;
    private Vector3 leftEyeGaze;
    private Vector3 rightEyeGaze;

    private Vector3 combinedEyeOrigin;
    private Vector3 combinedEyeGaze;

    public GameObject target;
    [SerializeField] private GameObject targetCylinder;
    public GameObject camObject;
    public GameObject XRRig;

    private Camera cam;
    private Vector3 targetGaze;

    [Header("Estimation Parameters:")]
    [SerializeField] private float fixationDuration = 0.5f;
    [SerializeField] private float thresholdAngle = 3f;

    private float timer;
    [HideInInspector] public bool targetFocused = false;
    [HideInInspector] public bool checkForFocus = false;

    private ExperimentManager expManager;

    void Start()
    {
        expManager = FindObjectOfType<ExperimentManager>();
        saveTracking = FindObjectOfType<SaveTracking>();
        cam = camObject.GetComponent<Camera>();

        ShowOutline();
    }

    public Vector2 GazeEstimate()
    {
        Vector2 target = new Vector2(0f, 0f);
        return target;
    }

    public void HideOutline()
    {
        targetCylinder.SetActive(false);
    }

    public void ShowOutline()
    {
        targetCylinder.SetActive(true);
    }

    public bool isLeft(Vector3 leftTargetPosition, Vector3 rightTargetPosition)
    {
        Vector3 origin = XRRig.transform.position;
        Vector3 up = Vector3.up;

        Vector3 right = rightTargetPosition - origin;
        Vector3 left = leftTargetPosition - origin;


        Debug.Log("angle = " + Vector3.SignedAngle(left, right, up));
        return Vector3.SignedAngle(left, right, up) > 0;
    }

    public bool isLeft2(Vector3 normal, Vector3 up, Vector3 right)
    {
        Debug.Log("angle = " + Vector3.SignedAngle(normal, up, right));
        return Vector3.SignedAngle(normal, up, right) < 0;
    }

    // Update is called once per frame
    void Update()
    {
        if (!expManager.conditionalTargetPresentation)
        {
            targetFocused = true;
            return;
        }

        saveTracking.GetGazeData();
        leftEyeOrigin = cam.transform.position + saveTracking.leftEyeOrigin;
        rightEyeOrigin = cam.transform.position + saveTracking.rightEyeOrigin;
        leftEyeGaze = cam.transform.rotation * saveTracking.leftEyeGaze;
        rightEyeGaze = cam.transform.rotation * saveTracking.rightEyeGaze;

        combinedEyeGaze = leftEyeGaze + rightEyeGaze;
        combinedEyeOrigin =  (leftEyeOrigin + rightEyeOrigin) / 2;

        targetGaze = target.transform.position - combinedEyeOrigin;
        Vector3 targetGazeLeft = target.transform.position - leftEyeOrigin;
        Vector3 targetGazeRight = target.transform.position - rightEyeOrigin;

        float angle = Vector3.Angle(combinedEyeGaze, targetGaze);
        float angleLeft = Vector3.Angle(leftEyeGaze, targetGazeLeft);
        float angleRight = Vector3.Angle(rightEyeGaze, targetGazeRight);

        if (checkForFocus)
        {
            if ((angleLeft > 0 && angleLeft < thresholdAngle) || (angleRight > 0 && angleRight < thresholdAngle)) //angle = 0 means no eye tracking data
            {
                timer += Time.deltaTime;
                ShowOutline();

            }            

            /*if (angle > 0 && angle < thresholdAngle) //angle = 0 means no eye tracking data
            {
                timer += Time.deltaTime;
                ShowOutline();

            }*/

            else
            {
                HideOutline();
            }

            // if the target was focused for a predefined amount of time, the experiment continues
            if (timer > fixationDuration)
            {
                targetFocused = true;
                checkForFocus = false;
                timer = 0f;
                ShowOutline();
            }
        }
    }
}
