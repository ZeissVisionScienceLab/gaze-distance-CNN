using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class PointManagerScript : MonoBehaviour
{
    [SerializeField] private TMP_Text pointText;
    [SerializeField] private GameObject floatingPoints;

    // private Manager manager;
    private Camera cam;
    private int oldPoints;
    private int actPoints;

    public int ActPoints
    {
        get { return actPoints; }
        set { actPoints = value; }
    }

    // Start is called before the first frame update
    void Start()
    {
        actPoints = 0;
        oldPoints = actPoints;
        cam = Camera.main;
        // manager = GameObject.Find("Manager").GetComponent<Manager>();
        
        UpdateText();
    }

    // Update is called once per frame
    void Update()
    {
        
        if (oldPoints != actPoints)
        {
            oldPoints = actPoints;
            UpdateText();
        }

    }

    private void UpdateText()
    {
        pointText.text = "Points: " + actPoints;
    }

    public void AddPoints(int value)
    {
        actPoints += value;
    }

    public void AddPoints(int value, Vector3 pos)
    {
        actPoints += value;
        GameObject myFloatingPoints = Instantiate(floatingPoints, pos, Quaternion.identity);
        myFloatingPoints.transform.forward = (pos - cam.gameObject.transform.position).normalized;
        if (value > 0)
        {
            myFloatingPoints.transform.GetChild(0).GetComponent<TextMesh>().color = Color.green;//manager.posPointsColor;
            //myFloatingPoints.transform.GetChild(0).GetComponent<TextMesh>().fontSize = 100*DistanceJoint2D;
            myFloatingPoints.transform.GetChild(0).GetComponent<TextMesh>().text = "+" + value.ToString();
        }
        else
        {
            myFloatingPoints.transform.GetChild(0).GetComponent<TextMesh>().color = Color.red;//manager.negPointsColor;
            myFloatingPoints.transform.GetChild(0).GetComponent<TextMesh>().text = value.ToString();
        }

    }
}
