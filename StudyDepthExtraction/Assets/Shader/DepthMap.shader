Shader "DepthData/DepthMap"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Gaze ("Gaze", Vector) = (0.5,0.5,0,0)
        _Radius ("Radius", Float) = 0.05
        _GazeColor ("GazeColor", Color) = (1,0,0,0)
        _ShowGaze ("ShowGaze", Float) = 1.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            sampler2D _MainTex, _CameraDepthTexture;
            float4 _MainTex_ST;
            float4 _Gaze;
            float _Radius;
            float4 _GazeColor;
            float _ShowGaze;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);

                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // sample the texture
                fixed4 col = tex2D(_MainTex, i.uv);

                float distFromGaze = distance(_Gaze.xy, i.uv.xy);

                if (distFromGaze < 0.01 && _ShowGaze > 0.5){ 
                    col.rgba = _GazeColor.rgba;
                }
                else
                {
                    float d = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv);  
                    float linearDepth = Linear01Depth(d);
                    d = linearDepth;
                    col = float4(d,d,d,1);
                }
                return col;
                
                
            }
            ENDCG
        }
    }
}
