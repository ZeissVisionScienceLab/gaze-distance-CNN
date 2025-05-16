using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
public class DepthRecording : MonoBehaviour
{
    public float nearClip = 0.25f;
    public float farClip = 15f;
    public bool saveDepth = false; // set true for start recording
    public int downsampleFactor = 4;
    private bool savingRightNow = false;
    private Camera cam;
    private Material depthMaterial; // material used for returning depth texture

    private int frameCounter;
    private int missedCounter;
    public int shaderPass = 2; // which pass of the depth shader should be used
    

    void Start()
    {
        cam = GetComponent<Camera>();
        //set camera to render depth
        cam.depthTextureMode |= DepthTextureMode.Depth;
        frameCounter = 0;
        missedCounter = 0;
        depthMaterial = new Material(Shader.Find("Hidden/DepthRecording"));

        // check if output folder exists
        if(!Directory.Exists(Application.dataPath + "/../recordings/"))
        {    
            //if it doesn't, create it
            Directory.CreateDirectory(Application.dataPath + "/../recordings/");
        }
    }

    // Use Update() to check if toggle key is pressed
    void Update()
    {
        if (Input.GetKeyDown("k"))
        {
            if(saveDepth)
            {
                saveDepth = false;
                Debug.Log("Stop recording depth data.");
            }
            else
            {
                saveDepth = true;
                Debug.Log("Start recording depth data.");
            }
        }
        if (Input.GetKeyDown("u"))
        {
            Debug.Log(frameCounter);
            Debug.Log(missedCounter);
        }

    }
    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (saveDepth && !savingRightNow) // save only if we are not already saving right now
        {
            //Start saving the depth texture
            StartCoroutine(SaveDepthTex(source));
        }
        else if(saveDepth)
        {
            missedCounter++;
        }
        Graphics.Blit(source, destination);
    }

    private IEnumerator SaveDepthTex(RenderTexture source)
    {
        savingRightNow = true;
        //RenderTexture depth = Shader.GetGlobalTexture ("_CameraDepthTexture") as RenderTexture;
        // create temporary render texture with reduced resolution
        int texWidth = source.width/downsampleFactor;
        int texHeight = source.height/downsampleFactor;
        RenderTexture tmp = RenderTexture.GetTemporary(texWidth, texHeight, 16, RenderTextureFormat.ARGBFloat);
        
        // set the min and far distance variables (value 0 in texture is nearClip, 1 is farClip)
        depthMaterial.SetFloat("_minDist", nearClip);
        depthMaterial.SetFloat("_farDist", farClip);
        Graphics.Blit(source, tmp, depthMaterial,shaderPass); // blit into tmp

        //Graphics.Blit(depth, tmp); // blit into tmp, then we can access depth buffer

        // Wait until the next frame
        yield return null;
        RenderTexture lastActive = RenderTexture.active;
        RenderTexture.active = tmp;
        //Copy the active render texture into a normal Texture2D
        //Unfortunately readpixels doesn't work with single channel formats, so RGBAFloat will have to do (32bit per channel?)
        Texture2D tex = new Texture2D(texWidth, texHeight, TextureFormat.RGBAFloat , false);
        tex.ReadPixels(new Rect(0, 0, texWidth, texHeight), 0, 0);
        tex.Apply();
        //Restore the active render texture and release our temporary tex
        RenderTexture.active = lastActive;
        RenderTexture.ReleaseTemporary(tmp);

        //Wait another frame
        yield return null;
        byte[] data = tex.EncodeToPNG(); // bit depth
        byte[] data2 = tex.EncodeToEXR();
        //Wait another frame
        yield return null;

        //Write the texture to a file - let's test png and exr (higher bit depth?)
        File.WriteAllBytes(Application.dataPath + "/../recordings/depthTexture" + frameCounter.ToString() + ".png", data);
        File.WriteAllBytes(Application.dataPath + "/../recordings/depthTexture" + frameCounter.ToString() + ".exr", data2);
        frameCounter++;
        savingRightNow = false;
    }
}