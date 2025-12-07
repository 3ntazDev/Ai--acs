"""
🚗 نظام تقييم حوادث السيارات بالذكاء الاصطناعي
Backend API كامل مع Gemini Vision AI - نسخة نهائية
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import google.generativeai as genai
from io import BytesIO
from PIL import Image
import json
from datetime import datetime
import os
import sys
from dotenv import load_dotenv

# تحميل متغيرات البيئة من ملف .env
load_dotenv()

# إصلاح مشكلة encoding في Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ================================
# 📊 نماذج البيانات
# ================================

class AnalysisResponse(BaseModel):
    incident_id: str
    severity_level: str
    severity_score: int
    damage_description: str
    injuries_detected: bool
    damaged_parts: List[str]
    vehicle_drivable: bool
    tow_needed: bool
    repair_cost: str
    recommended_action: str
    emergency_service: Optional[str]
    timestamp: str
    location: Optional[dict]
    technical_notes: Optional[str] = None

# ================================
# 🚀 تهيئة FastAPI
# ================================

app = FastAPI(
    title="Car Accident Analysis System",
    description="AI-powered car accident analysis using Gemini Vision",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# 🤖 إعداد Gemini AI
# ================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("⚠️ تحذير: لم يتم العثور على GEMINI_API_KEY")
    print("📝 قم بإنشاء ملف .env وأضف: GEMINI_API_KEY=your-api-key")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ تم تكوين Gemini API بنجاح")
    except Exception as e:
        print(f"❌ خطأ في تهيئة Gemini: {e}")

# استخدام موديل Gemini 2.5 Flash (يدعم الصور)
GEMINI_MODEL = "gemini-2.5-flash"

# ================================
# 🧠 وظيفة التحليل بـ Gemini Vision
# ================================

def analyze_accident_image(image_data: bytes) -> dict:
    """
    تحليل صورة الحادث باستخدام Gemini Vision AI
    """
    
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Gemini API Key not configured. Please add GEMINI_API_KEY to .env file"
        )
    
    try:
        # تحويل الصورة إلى PIL Image
        image = Image.open(BytesIO(image_data))
        
        # تحسين حجم الصورة إذا كانت كبيرة
        max_size = (1024, 1024)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read image: {str(e)}"
        )

    # Prompt للتحليل
    prompt = """You are an expert car accident assessor. Analyze this accident image and provide a JSON report.

    Return ONLY valid JSON with these exact keys:

    {
        "severity_score": <number 0-100>,
        "damage_description": "<detailed Arabic description of damage>",
        "injuries_detected": <true/false>,
        "damaged_parts": ["<list damaged parts in Arabic>"],
        "vehicle_drivable": <true/false>,
        "tow_needed": <true/false>,
        "repair_cost_level": "<منخفضة/متوسطة/عالية/عالية جداً>",
        "technical_notes": "<technical observations in Arabic>"
    }

    Severity scoring:
    - 0-30: Very minor (scratches)
    - 31-50: Minor damage
    - 51-70: Moderate damage
    - 71-85: Severe damage
    - 86-100: Critical damage

    Be accurate and thorough. Return ONLY the JSON object, no other text."""

    try:
        # إنشاء نموذج Gemini
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # إرسال الطلب
        response = model.generate_content(
            [prompt, image],
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                top_p=0.8,
                top_k=32,
            )
        )
        
        # استخراج النص
        response_text = response.text.strip()
        
        # تنظيف markdown
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # تحويل إلى JSON
        analysis_result = json.loads(response_text)
        
        # التحقق من المفاتيح المطلوبة
        required_keys = [
            "severity_score", "damage_description", "injuries_detected",
            "damaged_parts", "vehicle_drivable", "tow_needed", "repair_cost_level"
        ]
        
        for key in required_keys:
            if key not in analysis_result:
                raise ValueError(f"Missing required key: {key}")
        
        return analysis_result
        
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}")
        print(f"Response: {response_text[:500]}")
        raise HTTPException(
            status_code=500,
            detail="Failed to parse Gemini response as JSON"
        )
    except Exception as e:
        print(f"Analysis Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis error: {str(e)}"
        )

# ================================
# 🎯 تحديد الإجراء المناسب
# ================================

def determine_action(severity_score: int, injuries: bool) -> tuple:
    """تحديد الإجراء بناءً على الخطورة"""
    
    if severity_score >= 70 or injuries:
        return (
            "🚨 حادث خطير! تم إرسال إشعار لنجم (997). لا تحرك السيارة. تحقق من سلامة الركاب. انتظر الطوارئ.",
            "نجم (997)",
            "عالي"
        )
    elif severity_score >= 40:
        return (
            "⚠️ حادث متوسط. تم رفع بلاغ لأبشر. وثق من جميع الزوايا. تواصل مع التأمين.",
            "أبشر",
            "متوسط"
        )
    else:
        return (
            "✅ حادث بسيط. تم التوثيق. يمكنك التواصل مع التأمين. لا حاجة لإجراءات طارئة.",
            None,
            "منخفض"
        )

# ================================
# 📡 API Endpoints
# ================================

@app.get("/")
async def root():
    """الصفحة الرئيسية"""
    return {
        "message": "🚗 نظام تقييم حوادث السيارات بالذكاء الاصطناعي",
        "version": "2.0.0",
        "status": "✅ النظام يعمل",
        "ai_provider": "Google Gemini",
        "model": GEMINI_MODEL,
        "endpoints": {
            "analyze": "/analyze - تحليل صورة حادث",
            "health": "/health - فحص صحة النظام",
            "docs": "/docs - التوثيق التفاعلي"
        }
    }

@app.get("/health")
async def health_check():
    """فحص صحة النظام"""
    api_configured = bool(GEMINI_API_KEY)
    
    return {
        "status": "healthy" if api_configured else "warning",
        "timestamp": datetime.now().isoformat(),
        "ai_model": GEMINI_MODEL,
        "api_configured": api_configured,
        "message": "جاهز للتحليل" if api_configured else "يرجى تكوين GEMINI_API_KEY في ملف .env"
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_accident(
    file: UploadFile = File(..., description="صورة الحادث (JPEG, PNG, WebP)"),
    latitude: Optional[float] = Form(None, description="خط العرض"),
    longitude: Optional[float] = Form(None, description="خط الطول")
):
    """
    🎯 تحليل صورة حادث السيارة بالذكاء الاصطناعي
    """
    
    # التحقق من نوع الملف
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="نوع الملف غير مدعوم. الأنواع المسموحة: JPEG, PNG, WebP"
        )
    
    # الحد الأقصى لحجم الملف: 10MB
    max_size = 10 * 1024 * 1024
    
    try:
        # قراءة الصورة
        image_data = await file.read()
        
        if len(image_data) > max_size:
            raise HTTPException(
                status_code=400,
                detail="حجم الملف كبير جداً. الحد الأقصى 10MB"
            )
        
        # التحليل بالذكاء الاصطناعي
        print(f"🔍 بدء تحليل: {file.filename}")
        ai_analysis = analyze_accident_image(image_data)
        
        # تحديد الإجراء المناسب
        recommended_action, emergency_service, severity_level = determine_action(
            ai_analysis.get("severity_score", 0),
            ai_analysis.get("injuries_detected", False)
        )
        
        # معرف الحادث
        incident_id = f"ACC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # معلومات الموقع
        location_data = None
        if latitude is not None and longitude is not None:
            location_data = {
                "latitude": latitude,
                "longitude": longitude,
                "timestamp": datetime.now().isoformat()
            }
        
        # النتيجة النهائية
        result = AnalysisResponse(
            incident_id=incident_id,
            severity_level=severity_level,
            severity_score=ai_analysis.get("severity_score", 0),
            damage_description=ai_analysis.get("damage_description", ""),
            injuries_detected=ai_analysis.get("injuries_detected", False),
            damaged_parts=ai_analysis.get("damaged_parts", []),
            vehicle_drivable=ai_analysis.get("vehicle_drivable", True),
            tow_needed=ai_analysis.get("tow_needed", False),
            repair_cost=ai_analysis.get("repair_cost_level", "غير محدد"),
            recommended_action=recommended_action,
            emergency_service=emergency_service,
            timestamp=datetime.now().isoformat(),
            location=location_data,
            technical_notes=ai_analysis.get("technical_notes")
        )
        
        print(f"✅ تم التحليل بنجاح - {incident_id}")
        print(f"📊 الخطورة: {result.severity_score}/100 ({severity_level})")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ خطأ: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"خطأ في معالجة الطلب: {str(e)}"
        )

# ================================
# 🏃‍♂️ تشغيل التطبيق
# ================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 بدء تشغيل نظام تقييم الحوادث")
    print("=" * 60)
    print(f"🤖 AI Model: {GEMINI_MODEL}")
    print(f"🔑 API Key: {'✅ مُكوّن' if GEMINI_API_KEY else '❌ غير مُكوّن'}")
    print("=" * 60)
    print("📡 API متاح على: http://127.0.0.1:8000")
    print("📖 التوثيق متاح على: http://127.0.0.1:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )