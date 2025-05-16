using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using System.Threading;
using System.Text;
using System.Globalization;

public class SaveMetaData : MonoBehaviour
{
    Queue trackingDataQueue = new Queue();
    [HideInInspector] public String trackingFile = "metadata.csv";
    private ExperimentManager experimentManager;
    static string msgBuffer = "";

    // Start is called before the first frame update
    void Start()
    {
        
        experimentManager = FindObjectOfType<ExperimentManager>();
        trackingFile = experimentManager.metadataFilename;
        System.Threading.Thread.CurrentThread.CurrentCulture = new CultureInfo("de-DE");
        WriteHeader();
        Debug.Log("Start tracking meta data. Writing to " + trackingFile);
    }

    // Update is called once per frame
    void Update()
    {
       
    }

    public void Save()
    {
        var thread = new Thread(WriteData);
        thread.Start();
    }
    public void Msg(string msg)
    {
        msgBuffer = msg;
    }

    void WriteHeader()
    {
        StreamWriter sw = new StreamWriter(trackingFile);
        string header = "Timestamp;";
        header += "target_position.x;";
        header += "target_position.y;";
        header += "target_position.z;";
        header += "correct_answer;";
        header += "reaction_time;";
        header += "messages;";
        sw.WriteLine(header);
        sw.Close();
    }


    public void QueueData(Vector3 position, bool correctAnswer, float reactionTime)
    {
        StringBuilder datasetLine = new StringBuilder(600);
        datasetLine.Append(Time.time.ToString("F10") + ";");
        datasetLine.Append(position.x.ToString("F10") + ";");
        datasetLine.Append(position.y.ToString("F10") + ";");
        datasetLine.Append(position.z.ToString("F10") + ";");
        datasetLine.Append(correctAnswer.ToString() + ";");
        datasetLine.Append(reactionTime.ToString("F10") + ";");

        // buffered message
        if (!String.IsNullOrEmpty(msgBuffer))
        {
            datasetLine.Append(msgBuffer + ";");
            msgBuffer = "";
        }
        trackingDataQueue.Enqueue(datasetLine.ToString());

        trackingDataQueue.Enqueue(datasetLine.ToString());

    }

    public void WriteData()
    {
        Debug.Log("Writing Data");
        StreamWriter sw = new StreamWriter(trackingFile, true); //true for append
        string datasetLine;
        // dequeue trackingDataQueue until empty
        while (trackingDataQueue.Count > 0)
        {
            datasetLine = trackingDataQueue.Dequeue().ToString();
            sw.WriteLine(datasetLine); // write to file
        }
        sw.Close(); // close file
        Debug.Log("End Writing Data");
    }
}
