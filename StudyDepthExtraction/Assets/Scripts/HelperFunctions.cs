using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Text.RegularExpressions;
using System;

public class HelperFunctions : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public static (int sceneId, int participantId) ExtractSceneAndParticipantId(string fileName)
    {
        // Define the regex pattern for extracting scene and participant IDs
        string pattern = @"tracking_data_(\d+)_(\d+)\.csv";
        Regex regex = new Regex(pattern);

        // Match the pattern in the provided filename
        Match match = regex.Match(fileName);

        if (match.Success)
        {
            // Extract the scene and participant IDs from the match groups
            int sceneId = int.Parse(match.Groups[1].Value);
            int participantId = int.Parse(match.Groups[2].Value);

            return (sceneId, participantId);
        }
        else
        {
            throw new ArgumentException("Filename does not match the expected pattern.", nameof(fileName));
        }
    }
}
