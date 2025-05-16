Shader "Hidden/DepthRecording"
{
	Properties
	{
		_MainTex ("Texture", 2D) = "black" {}
	}

	CGINCLUDE
	#include "UnityCG.cginc"

	sampler2D _MainTex, _CameraDepthTexture;
	float _minDist, _farDist;

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
		o.uv = v.uv;
		return o;
	}
	ENDCG


	SubShader
	{
		// No culling or depth
		Cull Off ZWrite Off ZTest Always

		Pass // 0 direct depth pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			float4 frag (v2f i) : SV_Target
			{   
				float depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv);                				
				return float4(depth,0,0,1);
			}
			ENDCG
		}
		Pass // 1 RGB + Depth pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			float4 frag (v2f i) : SV_Target
			{
				float depth = Linear01Depth(SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv));
				float4 col = tex2D(_MainTex, i.uv);
				return float4(col.r,col.g,col.b,depth);
			}
			ENDCG
		}
		Pass // 2 linear01depth pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			float4 frag (v2f i) : SV_Target
			{
				float depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv);
				return float4(Linear01Depth(depth),Linear01Depth(depth),Linear01Depth(depth),1);
			}
			ENDCG
		}
		Pass // 3 linearEyeDepth pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			float4 frag (v2f i) : SV_Target
			{
				float depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv);
				return float4(LinearEyeDepth(depth),0,0,1);
			}
			ENDCG
		}
		Pass // 4 clamped pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			float4 frag (v2f i) : SV_Target
			{
				float depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv);
				float depth2 = (LinearEyeDepth(depth) - _minDist )/ (_farDist - _minDist);
				return float4(depth2,0,0,1);
			}
			ENDCG
		}
	}
}