using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ComputationManager : MonoBehaviour
{

    private float fov;
    private float pixelWidth;
    private float pixelHeight;
    public float kernelFieldOfView = 10f;
    private Camera extractDepthDataCamera;
    private ReplayManager replayManager;
    private ExtractDepthData extractDepthData;

    public int kernelHeight;


    // Start is called before the first frame update
    void Start()
    {
        replayManager = FindObjectOfType<ReplayManager>();
        extractDepthData = FindObjectOfType<ExtractDepthData>();

        extractDepthDataCamera = replayManager.depthExtractionCamera;

        // FoV is always vertical in Unity
        fov = extractDepthDataCamera.fieldOfView;
        Debug.Log("FoV: " + fov);
        pixelWidth = extractDepthData.GetWidth();
        pixelHeight = extractDepthData.GetHeight();

        kernelHeight = GetKernelHeightPixels();
        Debug.Log("Kernel Height in Pixels: " + kernelHeight);

    }


    public int GetKernelHeightPixels()
    {
        float focalLength = (pixelHeight / 2) / Mathf.Tan(fov / 2 * Mathf.Deg2Rad);
        float kernelHeight = 2 * focalLength * Mathf.Tan(kernelFieldOfView / 2 * Mathf.Deg2Rad);

        Debug.Log("Pixel Height in Pixels: " + pixelHeight);
        kernelHeight = pixelHeight / Mathf.Tan(fov / 2 * Mathf.Deg2Rad) * Mathf.Tan(kernelFieldOfView / 2 * Mathf.Deg2Rad);
        return (int) kernelHeight;

    }

    public Vector2Int ViewportToPixels(Vector3 viewportPosition)
    {
        Vector2Int pixelPosition = new Vector2Int();
        pixelPosition.x = (int) (viewportPosition.x * pixelWidth);
        pixelPosition.y = (int) (viewportPosition.y * pixelHeight);
        return pixelPosition;
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
