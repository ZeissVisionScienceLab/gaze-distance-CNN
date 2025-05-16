using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;

public class ExtractDepthData : MonoBehaviour
{
    public Material mat;

    // Shader options not necessary for this script
    private bool showGaze = false;
    private Color gazeColor = Color.red;

    public Camera extractDepthCamera;

    [Header("Render Texture Properties")]
    public int bitDepth = 16;

    [Header("Texture Resolution")]
    public int width = 1920;
    public int height = 1080;

    private int kernelDim;

    
    [Header("Depth Textures")]
    private RenderTexture renderTexture;
    public Texture2D texture2d;

    private float farClipPlane = 20f;
    private float nearClipPlane = 0.1f;

    private ComputationManager computationManager;
    private ReplayManager replayManager;

    private FileStream fileStream;
    private BinaryWriter binaryWriter;
    
    private string filePath;

    //public enum Scene {training, indoor, outdoor};



    //public string depthDataFilename = "depth_data.bin";


    // Start is called before the first frame update
    void Start()
    {

        computationManager = FindObjectOfType<ComputationManager>();
        replayManager = FindObjectOfType<ReplayManager>();

        // get relevant parameters from other scripts
        farClipPlane = extractDepthCamera.farClipPlane;
        nearClipPlane = extractDepthCamera.nearClipPlane;

        width = extractDepthCamera.pixelWidth;
        height = extractDepthCamera.pixelHeight;

        renderTexture = new RenderTexture(width, height, bitDepth, RenderTextureFormat.ARGBFloat);

        kernelDim = computationManager.kernelHeight;

        Debug.Log("(1) Kernel Height in Pixels: " + kernelDim);

        // set parameters
        mat.SetColor("_GazeColor", gazeColor);
        mat.SetFloat("_ShowGaze", showGaze ? 1 : 0);
    }

    public int GetWidth(){
        return extractDepthCamera.pixelWidth;
    }
    public int GetHeight(){
        return extractDepthCamera.pixelHeight;
    }
    public void Initialize(string depthDataFilename)
    {
        // Close and dispose the existing FileStream and BinaryWriter if they exist
        binaryWriter?.Close();
        fileStream?.Close();
        binaryWriter?.Dispose();
        fileStream?.Dispose();

        // Create the file path for the depth data file
        if (replayManager.scene == ReplayManager.Scene.indoor)
        {
            Debug.Log("Indoor");
            filePath = Application.dataPath + "/../resources/depthdata/indoor/" + depthDataFilename;
        }
        else if (replayManager.scene == ReplayManager.Scene.outdoor)
        {
            Debug.Log("Outdoor");
            filePath = Application.dataPath + "/../resources/depthdata/outdoor/" + depthDataFilename;
        }
        else
        {
            Debug.Log("Training");
            filePath = Application.dataPath + "/../resources/depthdata/training/" + depthDataFilename;
        }

        // Create a new FileStream and BinaryWriter with the new file path
        fileStream = new FileStream(filePath, FileMode.OpenOrCreate, FileAccess.Write);
        binaryWriter = new BinaryWriter(fileStream);
    }

    public void Dispose()
    {
        // Close and dispose the FileStream and BinaryWriter when done
        binaryWriter?.Close();
        fileStream?.Close();
        binaryWriter?.Dispose();
        fileStream?.Dispose();
    }

    public void Write(float data)
    {
        if (binaryWriter == null)
        {
            throw new InvalidOperationException("BinaryWriter is not initialized. Call Initialize() first.");
        }

        binaryWriter.Write(data);
        binaryWriter.Flush();
    }


    // Update is called once per frame
    void Update()
    {
    }

    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        Graphics.Blit(source, destination, mat);
    }

    public IEnumerator CameraToDepthTexture(Vector3 gaze){

        // get pixel position of combined eye gaze
        Vector2Int gazePixel = computationManager.ViewportToPixels(gaze);
        //Debug.Log("Pixel Gaze: " + gazePixel);
        //Debug.Log("Gaze Pixel: " + gazePixel);

        // render to render texture
        extractDepthCamera.targetTexture = renderTexture;
        texture2d = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGBAFloat, false);

        //Debug.Log("Render Texture Width: " + renderTexture.width);
        //Debug.Log("Render Texture Height: " + renderTexture.height);
        
        yield return new WaitForEndOfFrame();
        
        RenderTexture.active = renderTexture;

        // copy render texture to texture2d, write gpu data to cpu: computationally expensive!!
        texture2d.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        texture2d.Apply();

        // read gaze pixel from tex2d
        float c = texture2d.GetPixel(gazePixel.x, gazePixel.y).r * farClipPlane;// + nearClipPlane;
        //Debug.Log("Gaze depth: " + c);
        
        // extract kernel from texture2d
        //compute start coordinates of kernel
        
        //int startX = gazePixel.x - kernelDim / 2;
        //int startY = gazePixel.y - kernelDim / 2;

        //kernelDim = 200;
        int startX = gazePixel.x - kernelDim / 2;
        int startY = gazePixel.y - kernelDim / 2;

        //Debug.Log("(1)");
        //Debug.Log("StartX: " + startX);
        //Debug.Log("StartY: " + startY);

        // clamp start coordinates
        startX = Mathf.Clamp(startX, 0, renderTexture.width - kernelDim);
        startY = Mathf.Clamp(startY, 0, renderTexture.height - kernelDim);

        //Debug.Log("StartX: " + startX);
        //Debug.Log("StartY: " + startY);

        // retrieve kernel from texture2d
        Color[] kernelColors = new Color[kernelDim * kernelDim];
        kernelColors = texture2d.GetPixels(startX, startY, kernelDim, kernelDim, 0);

        // extract only r component of kernel colors
        float[] kernel = new float[kernelDim * kernelDim];
        float d = 0;
        float d1 = 0;
        float d2 = 0;

        //Debug.Log("kerneldim: " + kernelDim);
        
        for (int i = 0; i < kernelDim * kernelDim; i++)
        {
            // map 0-1 to nearClipPlane - farClipPlane
            d1 = nearClipPlane + (farClipPlane - nearClipPlane) * kernelColors[i].r;
            d2 = kernelColors[i].r * farClipPlane;// + nearClipPlane;

            kernel[i] = d2;

            // save to binary file
            Write(d2);
            
        }

        // release EVERYTHING, such that no memory problems appear
        
        RenderTexture.active = null;
        renderTexture.Release();
        Destroy(texture2d);

        extractDepthCamera.targetTexture = null;
    }


    /*
    public void Clear(){
        // Clean up and close the file when the object is destroyed
        if (binaryWriter != null)
        {
            binaryWriter.Close();
        }

        if (fileStream != null)
        {
            fileStream.Close();
        }
    }
*/
    void OnApplicationQuit()
    {
        // Clean up and close the file when the object is destroyed
        if (binaryWriter != null)
        {
            binaryWriter.Close();
        }

        if (fileStream != null)
        {
            fileStream.Close();
        }
    }
}
