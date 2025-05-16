using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ExecuteInEditMode]
public class PostProcessing : MonoBehaviour
{

    public Material mat;
    public bool showGaze = false;
    public Color gazeColor = Color.red;

    // Start is called before the first frame update
    void Start()
    {
        mat.SetColor("_GazeColor", gazeColor);
        mat.SetFloat("_ShowGaze", showGaze ? 1 : 0);
    }

    // Update is called once per frame
    void Update()
    {
    }

    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        Graphics.Blit(source, destination, mat);
    }
}
