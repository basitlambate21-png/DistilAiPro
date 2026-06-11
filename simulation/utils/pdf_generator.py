import io
from datetime import datetime

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet() if REPORTLAB_AVAILABLE else None

    def generate_mccabe_report(self, system, params, stages, R_min):
        if not REPORTLAB_AVAILABLE:
            return self._generate_simple_pdf("McCabe-Thiele Report", system, params)

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []

        # Title
        title = Paragraph(f"<b>McCabe-Thiele Report: {system}</b>", self.styles['Heading1'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))

        # Parameters
        data = [
            ['Parameter', 'Value'],
            ['Reflux Ratio (R)', f"{params['R']:.2f}"],
            ['Distillate Purity (xD)', f"{params['x_d']:.4f}"],
            ['Bottoms Purity (xB)', f"{params['x_b']:.4f}"],
            ['Feed Composition (zF)', f"{params['z_f']:.4f}"],
            ['Feed Condition (q)', f"{params['q']:.2f}"],
            ['Minimum Reflux (R_min)', f"{R_min:.2f}"],
            ['Total Stages', str(stages['n_stages'])],
            ['Feed Stage', str(stages['feed_stage'])]
        ]

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(table)
        doc.build(story)
        buffer.seek(0)
        return buffer, f"mccabe_report_{system}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    def generate_simulation_report(self, system, inputs, results):
        if not REPORTLAB_AVAILABLE:
            return self._generate_simple_pdf("Simulation Report", system, inputs)

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []

        title = Paragraph(f"<b>Process Simulation Report: {system}</b>", self.styles['Heading1'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))

        data = [
            ['Parameter', 'Value'],
            ['Reflux Ratio', f"{inputs['rr']:.2f}"],
            ['Feed Rate', f"{inputs['fr']:.1f} mol/hr"],
            ['Feed Composition', f"{inputs['fc']:.4f}"],
            ['Top Purity', f"{results['top_purity']:.4f}"],
            ['Bottom Purity', f"{results['bottom_purity']:.4f}"],
            ['Reboiler Duty', f"{results['reboiler_duty']:.1f} kW"],
            ['Condenser Duty', f"{results['condenser_duty']:.1f} kW"],
            ['Distillate Rate', f"{results['distillate_rate']:.1f} mol/hr"],
            ['Bottoms Rate', f"{results['bottoms_rate']:.1f} mol/hr"]
        ]

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(table)
        doc.build(story)
        buffer.seek(0)
        return buffer, f"simulation_report_{system}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    def generate_optimization_report(self, system, current, optimal, savings, savings_pct):
        if not REPORTLAB_AVAILABLE:
            return self._generate_simple_pdf("Optimization Report", system, current)

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []

        title = Paragraph(f"<b>AI Optimization Report: {system}</b>", self.styles['Heading1'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))

        data = [
            ['Parameter', 'Current', 'Optimized', 'Savings'],
            ['Reboiler Duty (kW)', f"{current['reboiler_duty']:.1f}", f"{optimal['reboiler_duty']:.1f}", f"{savings:.1f} ({savings_pct:.1f}%)"],
            ['Top Purity', f"{current['top_purity']:.4f}", f"{optimal['top_purity']:.4f}", "-"],
            ['Bottom Purity', f"{current['bottom_purity']:.4f}", f"{optimal['bottom_purity']:.4f}", "-"]
        ]

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(table)
        doc.build(story)
        buffer.seek(0)
        return buffer, f"optimization_report_{system}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    def _generate_simple_pdf(self, title_text, system, data):
        """Fallback when reportlab is not available"""
        buffer = io.BytesIO()
        buffer.write(f"{title_text}\n".encode())
        buffer.write(f"System: {system}\n".encode())
        buffer.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n".encode())
        buffer.write(f"Data: {str(data)}\n".encode())
        buffer.seek(0)
        return buffer, f"report_{system}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def pdf_download_button(buffer, filename, button_text):
    """Create a download button for PDF"""
    import streamlit as st
    import base64

    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()

    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">'
    href += f'<button style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; border-radius: 12px; padding: 0.75rem 2rem; font-weight: 600; cursor: pointer;">'
    href += f'📄 {button_text}'
    href += '</button></a>'

    st.markdown(href, unsafe_allow_html=True)
