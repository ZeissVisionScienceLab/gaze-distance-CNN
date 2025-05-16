using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraManager : MonoBehaviour
{

    public float speed = 10.0f;
    public float rotationSpeed = 5.0f;
    [Header("Scene 0")]
    public List<Vector3> camera_positions = new List<Vector3>();
    public List<Vector3> camera_rotations = new List<Vector3>();

    //public Camera cam;

    public GameObject XRRig;
    public GameObject secondCam;

    private int positionCounter = 0;

    // Start is called before the first frame update
    void Start()
    {
        //cam = GetComponent<Camera>();
        if (camera_positions.Count != camera_rotations.Count)
        {
            Debug.LogError("Camera position and rotation array lengths do not match. Please make sure they are of the same length.");
            return;
        }
    }

    // Update is called once per frame
    void Update()
    {
        //change camera positions with left and right arrow
        if (Input.GetKeyDown(KeyCode.RightArrow))
        {
            if(positionCounter <= camera_positions.Count){
                
                Debug.Log("Moving camera to position: " + camera_positions[positionCounter] + " with rotation: " + camera_rotations[positionCounter]);
                MoveCamera(camera_positions[positionCounter], camera_rotations[positionCounter]);
                positionCounter += 1;
            }
        }

        if (Input.GetKeyDown(KeyCode.LeftArrow))
        {
            if (positionCounter > 0)
            {
                positionCounter -= 1;
                Debug.Log("Moving camera to position: " + camera_positions[positionCounter] + " with rotation: " + camera_rotations[positionCounter]);
                MoveCamera(camera_positions[positionCounter], camera_rotations[positionCounter]);
            }
        }
        FlyCamera();
        if (Input.GetKeyDown(KeyCode.Return))
        {
            camera_positions.Add(secondCam.transform.position);
            camera_rotations.Add(secondCam.transform.rotation.eulerAngles);
            Debug.Log("Camera position saved.");
        }
    }


    private void FlyCamera()
    {   

        // Movement
        if (Input.GetKey(KeyCode.W))
        {
            secondCam.transform.position += secondCam.transform.forward * speed * Time.deltaTime;
        }
        if (Input.GetKey(KeyCode.S))
        {
            secondCam.transform.position -= secondCam.transform.forward * speed * Time.deltaTime;
        }
        if (Input.GetKey(KeyCode.A))
        {
            secondCam.transform.position -= secondCam.transform.right * speed * Time.deltaTime;
        }
        if (Input.GetKey(KeyCode.D))
        {
            secondCam.transform.position += secondCam.transform.right * speed * Time.deltaTime;
        }
        if (Input.GetKey(KeyCode.Q))
        {
            secondCam.transform.position -= secondCam.transform.up * speed * Time.deltaTime;
        }
        if (Input.GetKey(KeyCode.E))
        {
            secondCam.transform.position += secondCam.transform.up * speed * Time.deltaTime;
        }

        // Rotation
        if (Input.GetMouseButton(1)) // Right mouse button
        {
            float h = rotationSpeed * Input.GetAxis("Mouse X");
            float v = rotationSpeed * Input.GetAxis("Mouse Y");

            secondCam.transform.Rotate(-v, h, 0);

            // Remove roll
            secondCam.transform.rotation = Quaternion.Euler(secondCam.transform.rotation.eulerAngles.x, secondCam.transform.rotation.eulerAngles.y, 0);
        }
    }

    private void MoveCamera(Vector3 pos, Vector3 rotation)
    {
        secondCam.transform.position = pos;
        secondCam.transform.rotation = Quaternion.Euler(rotation);

        //transform.eulerAngles = rotation;
    }
}
