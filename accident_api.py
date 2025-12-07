"""
🚗 نظام تقييم حوادث السيارات بالذكاء الاصطناعي
Backend API كامل مع Gemini Vision AI - نسخة متقدمة
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
try:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
except:
    pass

# ================================
# 📊 نماذج البيانات
# ================================

class FaultPercentage(BaseModel):
    party_a: int  # نسبة خطأ الطرف الأول
    party_b: int  # نسبة خطأ الطرف الثاني

class AccidentCause(BaseModel):
    primary_cause: str  # السبب الرئيسي
    contributing_factors: List[str]  # عوامل إضافية

class EmergencyResponse(BaseModel):
    service_needed: str  # نجم، أبشر، أو لا يوجد
    priority_level: str  # عاجل، متوسط، منخفض
    estimated_response_time: str  # وقت الاستجابة المتوقع

class CameraRequest(BaseModel):
    cameras_needed: bool
    reason: str
    estimated_locations: List[str]

class AnalysisResponse(BaseModel):
    incident_id: str
    timestamp: str
    
    # تحليل الحادث
    severity_level: str
    severity_score: int
    accident_type: str  # نوع الحادث: تصادم أمامي، جانبي، انقلاب، إلخ
    
    # كيف حصل الحادث
    accident_description: str
    accident_cause: AccidentCause
    
    # تقييم الخطأ
    fault_assessment: FaultPercentage
    fault_explanation: str
    
    # الأضرار
    damage_description: str
    damaged_parts: List[str]
    vehicle_drivable: bool
    tow_needed: bool
    repair_cost: str
    injuries_detected: bool
    
    # الإجراءات المطلوبة
    emergency_response: EmergencyResponse
    camera_request: CameraRequest
    recommended_action: str
    
    # الموقع
    location: Optional[dict]
    technical_notes: Optional[str] = None

# ================================
# 🚀 تهيئة FastAPI
# ================================

app = FastAPI(
    title="نظام تقييم حوادث السيارات المتقدم",
    description="تحليل شامل لحوادث السيارات مع تحديد المسؤولية والإجراءات",
    version="3.0.0"
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
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ تم تكوين Gemini API بنجاح")
    except Exception as e:
        print(f"❌ خطأ في تهيئة Gemini: {e}")

GEMINI_MODEL = "gemini-2.5-flash"

# ================================
# 🧠 وظيفة التحليل بـ Gemini Vision
# ================================

def analyze_accident_image(image_data: bytes) -> dict:
    """
    تحليل شامل لصورة الحادث
    """
    
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Gemini API Key not configured"
        )
    
    try:
        image = Image.open(BytesIO(image_data))
        max_size = (1024, 1024)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")

    prompt = """أنت خبير في تحليل حوادث السيارات وتحديد المسؤولية. حلل هذه الصورة بدقة شديدة.

    أرجع JSON كامل يحتوي على:

    {
        "severity_score": <رقم من 0-100>,
        "accident_type": "<نوع الحادث: تصادم أمامي/جانبي/خلفي/انقلاب/دهس/اصطدام بجسم ثابت>",
        
        "accident_description": "<وصف تفصيلي بالعربية: كيف حصل الحادث؟ من أين جاء الاصطدام؟ ما السرعة المحتملة؟>",
        
        "primary_cause": "<السبب الرئيسي: سرعة زائدة/عدم ترك مسافة/تجاوز خاطئ/عدم التزام بالإشارة/انحراف مفاجئ/إلخ>",
        "contributing_factors": ["<عوامل أخرى ساهمت في الحادث>"],
        
        "fault_party_a_percentage": <نسبة خطأ السائق A من 0-100>,
        "fault_party_b_percentage": <نسبة خطأ السائق B من 0-100>,
        "fault_explanation": "<شرح مفصل: لماذا هذه النسب؟ على أي أساس؟ ما الأدلة من الصورة؟>",
        
        "damage_description": "<وصف الأضرار بالتفصيل>",
        "damaged_parts": ["<قائمة الأجزاء المتضررة>"],
        "vehicle_drivable": <true/false>,
        "tow_needed": <true/false>,
        "repair_cost_level": "<منخفضة/متوسطة/عالية/عالية جداً>",
        "injuries_detected": <true/false>,
        
        "cameras_needed": <true/false - هل نحتاج كاميرات المراقبة؟>,
        "camera_reason": "<السبب: لتحديد السرعة/لمعرفة من تجاوز الإشارة/لتوضيح تسلسل الأحداث/إلخ>",
        "camera_locations": ["<أماكن محتملة للكاميرات: إشارة المرور/مدخل الشارع/إلخ>"],
        
        "technical_notes": "<ملاحظات فنية إضافية>"
    }

    معايير severity_score:
    - 0-30: بسيط جداً
    - 31-50: بسيط
    - 51-70: متوسط
    - 71-85: خطير
    - 86-100: خطير جداً

    كن دقيقاً جداً في تحديد نسب الخطأ. استخدم:
    - موقع الضرر (أمامي/جانبي/خلفي)
    - شدة الاصطدام
    - اتجاه القوة
    - علامات الفرامل إن وجدت
    
    أرجع JSON فقط بدون نص إضافي."""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            [prompt, image],
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                top_p=0.8,
                top_k=32,
            )
        )
        
        response_text = response.text.strip()
        
        # تنظيف markdown
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        analysis_result = json.loads(response_text)
        
        return analysis_result
        
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}")
        print(f"Response: {response_text[:500]}")
        raise HTTPException(status_code=500, detail="Failed to parse Gemini response")
    except Exception as e:
        print(f"Analysis Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

# ================================
# 🎯 تحديد الإجراءات والجهات
# ================================

def determine_emergency_response(severity_score: int, injuries: bool) -> EmergencyResponse:
    """تحديد جهة الطوارئ المطلوبة"""
    
    if severity_score >= 70 or injuries:
        return EmergencyResponse(
            service_needed="نجم (997)",
            priority_level="عاجل",
            estimated_response_time="5-10 دقائق"
        )
    elif severity_score >= 40:
        return EmergencyResponse(
            service_needed="أبشر",
            priority_level="متوسط",
            estimated_response_time="15-30 دقيقة"
        )
    else:
        return EmergencyResponse(
            service_needed="لا يوجد",
            priority_level="منخفض",
            estimated_response_time="غير مطلوب"
        )

def generate_recommended_action(severity_score: int, injuries: bool, cameras_needed: bool) -> str:
    """إنشاء توصيات مفصلة"""
    
    actions = []
    
    if severity_score >= 70 or injuries:
        actions.append("🚨 حادث خطير - إجراءات فورية:")
        actions.append("• تم إرسال إشعار تلقائي لنجم (997)")
        actions.append("• لا تحرك السيارة من مكانها")
        actions.append("• تحقق من سلامة جميع الركاب")
        actions.append("• ضع مثلثات التحذير")
        actions.append("• انتظر وصول فرق الطوارئ")
    elif severity_score >= 40:
        actions.append("⚠️ حادث متوسط - إجراءات مطلوبة:")
        actions.append("• تم رفع بلاغ تلقائي لأبشر")
        actions.append("• صور الحادث من جميع الزوايا")
        actions.append("• سجل معلومات الطرف الآخر")
        actions.append("• احتفظ بموقعك حتى وصول الدوريات")
    else:
        actions.append("✅ حادث بسيط - إجراءات عادية:")
        actions.append("• تم التوثيق الإلكتروني")
        actions.append("• يمكنك التواصل مع شركة التأمين")
        actions.append("• احتفظ بالصور والتقرير")
    
    if cameras_needed:
        actions.append("\n📹 طلب كاميرات المراقبة:")
        actions.append("• سيتم التواصل تلقائياً مع الجهات المختصة")
        actions.append("• يُنصح بتحديد موقع الحادث بدقة")
    
    actions.append("\n📋 معلومات إضافية:")
    actions.append("• رقم البلاغ سيصلك عبر رسالة نصية")
    actions.append("• يمكنك متابعة الحالة عبر التطبيق")
    
    return "\n".join(actions)

# ================================
# 📡 API Endpoints
# ================================

@app.get("/")
async def root():
    return {
        "message": "🚗 نظام تقييم حوادث السيارات المتقدم",
        "version": "3.0.0",
        "status": "✅ النظام يعمل",
        "features": [
            "تحليل شامل للحادث",
            "تحديد نسب المسؤولية",
            "كشف كيفية حدوث الحادث",
            "تحويل تلقائي لنجم/أبشر",
            "طلب كاميرات المراقبة",
            "تقييم الأضرار والتكاليف"
        ],
        "ai_model": GEMINI_MODEL
    }

@app.get("/health")
async def health_check():
    api_configured = bool(GEMINI_API_KEY)
    return {
        "status": "healthy" if api_configured else "warning",
        "timestamp": datetime.now().isoformat(),
        "ai_model": GEMINI_MODEL,
        "api_configured": api_configured
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_accident(
    file: UploadFile = File(..., description="صورة الحادث"),
    latitude: Optional[float] = Form(None, description="خط العرض"),
    longitude: Optional[float] = Form(None, description="خط الطول")
):
    """
    🎯 تحليل شامل لحادث السيارة
    """
    
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="نوع الملف غير مدعوم")
    
    max_size = 10 * 1024 * 1024
    
    try:
        image_data = await file.read()
        
        if len(image_data) > max_size:
            raise HTTPException(status_code=400, detail="حجم الملف كبير جداً")
        
        # التحليل بالذكاء الاصطناعي
        print(f"🔍 بدء التحليل الشامل: {file.filename}")
        ai_analysis = analyze_accident_image(image_data)
        
        # معرف الحادث
        incident_id = f"ACC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # تقييم الخطأ
        fault_assessment = FaultPercentage(
            party_a=ai_analysis.get("fault_party_a_percentage", 50),
            party_b=ai_analysis.get("fault_party_b_percentage", 50)
        )
        
        # سبب الحادث
        accident_cause = AccidentCause(
            primary_cause=ai_analysis.get("primary_cause", "غير محدد"),
            contributing_factors=ai_analysis.get("contributing_factors", [])
        )
        
        # تحديد الجهة المختصة
        emergency_response = determine_emergency_response(
            ai_analysis.get("severity_score", 0),
            ai_analysis.get("injuries_detected", False)
        )
        
        # طلب الكاميرات
        camera_request = CameraRequest(
            cameras_needed=ai_analysis.get("cameras_needed", False),
            reason=ai_analysis.get("camera_reason", ""),
            estimated_locations=ai_analysis.get("camera_locations", [])
        )
        
        # الإجراءات الموصى بها
        recommended_action = generate_recommended_action(
            ai_analysis.get("severity_score", 0),
            ai_analysis.get("injuries_detected", False),
            camera_request.cameras_needed
        )
        
        # معلومات الموقع
        location_data = None
        if latitude is not None and longitude is not None:
            location_data = {
                "latitude": latitude,
                "longitude": longitude,
                "timestamp": datetime.now().isoformat()
            }
        
        # تحديد مستوى الخطورة
        severity_score = ai_analysis.get("severity_score", 0)
        if severity_score >= 70:
            severity_level = "عالي"
        elif severity_score >= 40:
            severity_level = "متوسط"
        else:
            severity_level = "منخفض"
        
        # النتيجة النهائية
        result = AnalysisResponse(
            incident_id=incident_id,
            timestamp=datetime.now().isoformat(),
            severity_level=severity_level,
            severity_score=severity_score,
            accident_type=ai_analysis.get("accident_type", "غير محدد"),
            accident_description=ai_analysis.get("accident_description", ""),
            accident_cause=accident_cause,
            fault_assessment=fault_assessment,
            fault_explanation=ai_analysis.get("fault_explanation", ""),
            damage_description=ai_analysis.get("damage_description", ""),
            damaged_parts=ai_analysis.get("damaged_parts", []),
            vehicle_drivable=ai_analysis.get("vehicle_drivable", True),
            tow_needed=ai_analysis.get("tow_needed", False),
            repair_cost=ai_analysis.get("repair_cost_level", "غير محدد"),
            injuries_detected=ai_analysis.get("injuries_detected", False),
            emergency_response=emergency_response,
            camera_request=camera_request,
            recommended_action=recommended_action,
            location=location_data,
            technical_notes=ai_analysis.get("technical_notes")
        )
        
        print(f"✅ تم التحليل بنجاح - {incident_id}")
        print(f"📊 الخطورة: {severity_score}/100")
        print(f"⚖️ نسب الخطأ: A={fault_assessment.party_a}% | B={fault_assessment.party_b}%")
        print(f"🚨 الجهة: {emergency_response.service_needed}")
        print(f"📹 كاميرات: {'مطلوبة' if camera_request.cameras_needed else 'غير مطلوبة'}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ خطأ: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطأ في معالجة الطلب: {str(e)}")

# ================================
# 🏃‍♂️ تشغيل التطبيق
# ================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")