using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;



public class PlaceTargets : MonoBehaviour
{
    // create a list of 3d vectors
    public List<Vector3> target_positions = new List<Vector3>();
    private List<Vector3> target_positions_acc = new List<Vector3>();
    private List<Vector3> target_normals = new List<Vector3>();
    private List<Vector3> target_normals_acc = new List<Vector3>();


    public List<Vector3> camera_positions = new List<Vector3>();
    private List<Vector3> camera_rotations = new List<Vector3>();
    
    public GameObject XRRig;
    public GameObject camObject;
    public GameObject secondCam;
    public GameObject camOffset;

    [HideInInspector] public String filename = "tracking_positions.csv";
    private Camera cam;

    public float distanceThreshold = 0.35f;
    private ExperimentManager experimentManager;

    void Start()
    {
        cam = secondCam.GetComponent<Camera>(); //camObject.GetComponent<Camera>();
        System.Globalization.CultureInfo customCulture = (System.Globalization.CultureInfo)System.Threading.Thread.CurrentThread.CurrentCulture.Clone();
        customCulture.NumberFormat.NumberDecimalSeparator = ".";

        System.Threading.Thread.CurrentThread.CurrentCulture = customCulture;
        experimentManager = FindObjectOfType<ExperimentManager>();
        filename = experimentManager.targetsFilename;
    }

    // Update is called once per frame
    void Update()
    {

        if (Input.GetMouseButtonDown(0))
        {
            Vector3 target_pos = Input.mousePosition;
            Debug.Log("cam pos:" + cam.transform.position + "cam rot:" + cam.transform.rotation);
            Ray ray = cam.ScreenPointToRay(target_pos);
            Debug.Log("Ray: " + ray);

            // find first intersection point of ray with any object of the scene
            RaycastHit hit;
            if (Physics.Raycast(ray, out hit))
            {
                Debug.Log("We hit something");
                Debug.Log("Hit: " + hit.point);
                target_positions_acc.Add(hit.point);

                // create a sphere at the intersection point
                Vector3 hit_point = hit.point;
                Debug.Log("hit point:" + hit_point.x + "," + hit_point.y + "," + hit_point.z);

                // get distance to object
                float distance = Vector3.Distance(hit_point , secondCam.transform.position);
                Debug.Log("Distance: " + distance);
                if(distance < distanceThreshold)
                    Debug.Log("Target to close! Distance = " + distance);

                // scale the sphere according to the distance
                float baseSize = 0.01f;
                //float x = (float) (baseSize * distance);
                //float z = (float) (baseSize * distance);
                float y = 0.0005f;// (float) (baseSize * distance);

                // visualize a flat cylinder at the hit point
                GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                Destroy(cylinder.GetComponent<CapsuleCollider>());
                cylinder.transform.position = hit_point;
                

                // rotate cylinder to align with the normal of the hit point
                Vector3 normal = hit.normal;
                target_normals_acc.Add(normal);

                Vector3 up = new Vector3(0, 1, 0);
                Vector3 axis = Vector3.Cross(up, normal);
                float angle = Vector3.Angle(up, normal);
                

                // scale x and z coordinates of the cylinder such that it looks the same from each direction, consider ray direction and normal of the object
                Vector2 a = new Vector2((-1)*normal.x, (-1)*normal.z);
                Vector2 b = new Vector2(ray.direction.x, ray.direction.z);
                float angle_xz = Vector2.Angle(a, b);
                float x_scaled = baseSize / Mathf.Cos(angle_xz * Mathf.Deg2Rad) * distance;

                Vector2 c = new Vector2((-1)*normal.y, (-1)*normal.z);
                Vector2 d = new Vector2(ray.direction.y, ray.direction.z);
                float angle_yz = Vector2.Angle(c, d);
                float z_scaled = baseSize / Mathf.Cos(angle_yz * Mathf.Deg2Rad) * distance;



                Debug.Log("Angle x/z: " + angle_xz + ", old_x = " + baseSize * distance + ", new x = " + x_scaled);
                Debug.Log("Angle y/z: " + angle_yz + ", old_y = " + baseSize * distance + ", new y = " + z_scaled);

                cylinder.transform.localScale = new Vector3(baseSize * distance, y, baseSize * distance);
                cylinder.transform.Rotate(axis, angle);
                cylinder.GetComponent<Renderer>().material.color = Color.magenta;
            }
        }

        if (Input.GetKeyDown("space"))
        {
            if (target_positions_acc.Count > 0)
            {
                Vector3 tPos = target_positions_acc[target_positions_acc.Count - 1];
                Vector3 cPos = secondCam.transform.position;

                float dist = Vector3.Distance(cPos, tPos);
                if(dist > distanceThreshold) {
                    // add last element added to target_positions_acc to target_positions
                    target_positions.Add(target_positions_acc[target_positions_acc.Count - 1]);
                    target_normals.Add(target_normals_acc[target_normals_acc.Count - 1]);
                    camera_positions.Add(secondCam.transform.position);
                    camera_rotations.Add(secondCam.transform.eulerAngles);
                    Debug.Log("Target added at: " + target_positions[target_positions.Count - 1]);
                }
                else
                {
                    Debug.Log("Target to close! Distance = " + dist);
                }

                
            }else{
                Debug.Log("No target positions added yet.");
            }
        }

        if (Input.GetKeyDown("backspace"))
        {
            if (target_positions.Count > 0)
            {
                // add last element added to target_positions_acc to target_positions
                target_positions.RemoveAt(target_positions.Count - 1);
                target_normals.RemoveAt(target_positions.Count - 1);
                camera_positions.RemoveAt(target_positions.Count - 1);
                camera_rotations.RemoveAt(target_positions.Count - 1);
                Debug.Log("Last target removed");
            }
            else
            {
                Debug.Log("No target positions added yet.");
            }
        }


    }

    void OnApplicationQuit()
    {
        SaveVectorListToCSV(filename);
    }

    private void SaveVectorListToCSV(string fileName)
    {
        string path = fileName;

        // save the list of vectors to a csv file, use commas as decimal separator and ; as column separator
        using (StreamWriter writer = new StreamWriter(path, false))
        {
            // write header
            writer.WriteLine("x;y;z;normal_x;normal_y;normal_z;camera_posx;camera_posy;camera_posz;camera_rotx;camera_roty;camera_rotz");
            
            for(int i = 0; i < target_positions.Count; i++){
                Vector3 v = target_positions[i];
                Vector3 n = target_normals[i];
                Vector3 cam_pos = camera_positions[i];
                Vector3 cam_rot = camera_rotations[i];

                writer.WriteLine(v.x.ToString() + ";" + v.y.ToString() + ";" + v.z.ToString() + ";" + n.x.ToString() + ";" + n.y.ToString() + ";" + n.z.ToString() + ";" + cam_pos.x.ToString() + ";" + cam_pos.y.ToString() + ";" + cam_pos.z.ToString() + ";" + cam_rot.x.ToString() + ";" + cam_rot.y.ToString() + ";" + cam_rot.z.ToString());
            }
        }

        Debug.Log("Vector list saved to: " + path);
    }
}
