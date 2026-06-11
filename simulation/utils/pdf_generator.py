from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.enums import TA_CENTER
import io
import base64
import datetime
import streamlit as st

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        self.section_style = ParagraphStyle(
            'CustomSection',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#764ba2'),
            spaceAfter=12,
            spaceBefore=12
        )
        self.subsection_style = ParagraphStyle(
            'CustomSubSection',
            parent=self.styles['Heading3'],
            fontSize=13,
            textColor=colors.HexColor('#1e1e2e'),
            spaceAfter=8,
            spaceBefore=8
        )
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14
        )
        self.info_style = ParagraphStyle(
            'InfoStyle',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#6b7280'),
            leading=12
        )
        self.bullet_style = ParagraphStyle(
            'BulletStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=16,
            leftIndent=20,
            bulletIndent=10,
            bulletFontName='Helvetica-Bold'
        )
    
    def create_header_footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#6b7280'))
        canvas.drawString(72, 40, f"DistilAI Pro Report | Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        canvas.drawRightString(520, 40, f"Page {doc.page}")
        canvas.restoreState()
    
    def generate_simulation_report(self, system_name, inputs, results, filename=None):
        if filename is None:
            filename = f"simulation_report_{system_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=72, bottomMargin=72)
        story = []
        
        story.append(Paragraph("DistilAI Pro - Process Simulation Report", self.title_style))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("System Information", self.section_style))
        story.append(Paragraph(f"System: {system_name.replace('-', ' ').title()}", self.normal_style))
        story.append(Paragraph(f"Report Date: {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}", self.normal_style))
        story.append(Spacer(1, 15))
        
        story.append(Paragraph("Input Parameters", self.section_style))
        input_data = [
            ['Parameter', 'Value', 'Unit'],
            ['Number of Stages', str(inputs.get('n_stages', 20)), '-'],
            ['Feed Stage', str(inputs.get('feed_stage', 10)), '-'],
            ['Pressure', str(inputs.get('pressure', 101.325)), 'kPa'],
            ['Reflux Ratio', f"{inputs.get('rr', 3.0):.2f}", '-'],
            ['Feed Rate', f"{inputs.get('fr', 100.0):.1f}", 'mol/hr'],
            ['Feed Composition', f"{inputs.get('fc', 0.5):.3f}", 'mole fraction'],
        ]
        
        input_table = Table(input_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
        input_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f3f4f6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f9fafb'), colors.HexColor('#ffffff')]),
        ]))
        story.append(input_table)
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("Simulation Results", self.section_style))
        result_data = [
            ['Result', 'Value', 'Unit'],
            ['Top Purity (xD)', f"{results.get('top_purity', 0):.4f}", 'mole fraction'],
            ['Bottom Purity (xB)', f"{results.get('bottom_purity', 0):.4f}", 'mole fraction'],
            ['Reboiler Duty', f"{results.get('reboiler_duty', 0):.2f}", 'kW'],
            ['Condenser Duty', f"{results.get('condenser_duty', 0):.2f}", 'kW'],
            ['Distillate Rate', f"{results.get('distillate_rate', 0):.2f}", 'mol/hr'],
            ['Bottoms Rate', f"{results.get('bottoms_rate', 0):.2f}", 'mol/hr'],
            ['Top Temperature', f"{results.get('top_temperature', 0):.2f}", '°C'],
            ['Bottom Temperature', f"{results.get('bottom_temperature', 0):.2f}", '°C'],
            ['Feed Temperature', f"{results.get('feed_temperature', 0):.2f}", '°C'],
            ['Relative Volatility', f"{results.get('relative_volatility', 0):.2f}", '-'],
        ]
        
        result_table = Table(result_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10b981')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0fdf4'), colors.HexColor('#ffffff')]),
        ]))
        story.append(result_table)
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("Generated by DistilAI Pro - AI-Powered Distillation Optimization", self.info_style))
        
        doc.build(story, onFirstPage=self.create_header_footer, onLaterPages=self.create_header_footer)
        buffer.seek(0)
        return buffer, filename
    
    def generate_mccabe_report(self, system_name, params, stages_data, R_min, filename=None):
        if filename is None:
            filename = f"mccabe_report_{system_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=72, bottomMargin=72)
        story = []
        
        story.append(Paragraph("DistilAI Pro - McCabe-Thiele Design Report", self.title_style))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("System Information", self.section_style))
        story.append(Paragraph(f"System: {system_name.replace('-', ' ').title()}", self.normal_style))
        story.append(Paragraph(f"Report Date: {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}", self.normal_style))
        story.append(Spacer(1, 15))
        
        story.append(Paragraph("Design Parameters", self.section_style))
        param_data = [
            ['Parameter', 'Value', 'Description'],
            ['xD (Distillate)', f"{params.get('x_d', 0.95):.3f}", 'Desired top product purity'],
            ['xB (Bottoms)', f"{params.get('x_b', 0.05):.3f}", 'Desired bottom product purity'],
            ['zF (Feed)', f"{params.get('z_f', 0.5):.3f}", 'Feed composition'],
            ['R (Reflux Ratio)', f"{params.get('R', 3.0):.2f}", 'Operating reflux ratio'],
            ['q (Feed Condition)', f"{params.get('q', 1.0):.2f}", '1.0=sat.liquid, 0.0=sat.vapor'],
            ['Rmin', f"{R_min:.2f}", 'Minimum reflux ratio'],
            ['R/Rmin', f"{params.get('R', 3.0)/R_min:.2f}" if R_min > 0 else "N/A", 'Operating ratio'],
        ]
        
        param_table = Table(param_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        param_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f3ff'), colors.HexColor('#ffffff')]),
        ]))
        story.append(param_table)
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("Design Results", self.section_style))
        n_stages = stages_data.get('n_stages', 0)
        feed_stage = stages_data.get('feed_stage', 0)
        story.append(Paragraph(f"Total Number of Stages: {n_stages}", self.normal_style))
        story.append(Paragraph(f"Feed Stage Location: Stage {feed_stage}", self.normal_style))
        story.append(Spacer(1, 15))
        
        story.append(Paragraph("Stage Compositions", self.subsection_style))
        
        stage_table_data = [['Stage', 'x (Liquid)', 'y (Vapor)', 'Type']]
        
        stages_x = stages_data.get('stages_x', [])
        stages_y = stages_data.get('stages_y', [])
        
        for i in range(0, len(stages_x), 2):
            stage_num = i // 2 + 1
            if stage_num > 50:
                break
                
            x_val = stages_x[i] if i < len(stages_x) else 0
            y_val = stages_y[i] if i < len(stages_y) else 0
            stage_type = 'FEED' if stage_num == feed_stage else 'Stage'
            
            stage_table_data.append([
                str(stage_num),
                f"{x_val:.4f}",
                f"{y_val:.4f}",
                stage_type
            ])
        
        if len(stage_table_data) > 22:
            stage_table_data = stage_table_data[:22]
            stage_table_data.append(['...', '...', '...', '...'])
        
        stage_table = Table(stage_table_data, colWidths=[1*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        stage_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f59e0b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#fffbeb'), colors.HexColor('#ffffff')]),
        ]))
        story.append(stage_table)
        story.append(Spacer(1, 20))
        
        # FIXED: Use bullet list instead of <br> tags
        story.append(Paragraph("Design Notes", self.subsection_style))
        
        notes = [
            "The McCabe-Thiele method assumes constant molar overflow and equilibrium stages.",
            "Actual column may require more stages due to stage efficiency.",
            "Feed stage location is approximate and should be verified with rigorous simulation.",
            "For azeotropic systems, purity may be limited by the azeotrope composition."
        ]
        
        for note in notes:
            story.append(Paragraph(f"• {note}", self.bullet_style))
        
        story.append(Spacer(1, 20))
        story.append(Paragraph("Generated by DistilAI Pro", self.info_style))
        
        doc.build(story, onFirstPage=self.create_header_footer, onLaterPages=self.create_header_footer)
        buffer.seek(0)
        return buffer, filename
    
    def generate_optimization_report(self, system_name, current, optimal, savings, savings_pct, filename=None):
        if filename is None:
            filename = f"optimization_report_{system_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=72, bottomMargin=72)
        story = []
        
        story.append(Paragraph("DistilAI Pro - AI Optimization Report", self.title_style))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("System Information", self.section_style))
        story.append(Paragraph(f"System: {system_name.replace('-', ' ').title()}", self.normal_style))
        story.append(Paragraph(f"Report Date: {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}", self.normal_style))
        story.append(Spacer(1, 15))
        
        story.append(Paragraph("Optimization Summary", self.section_style))
        summary_data = [
            ['Metric', 'Value'],
            ['Energy Savings', f"{savings:.2f} kW"],
            ['Savings Percentage', f"{savings_pct:.1f}%"],
            ['Current Reboiler Duty', f"{current.get('reboiler_duty', 0):.2f} kW"],
            ['Optimized Reboiler Duty', f"{optimal.get('reboiler_duty', 0):.2f} kW"],
            ['Current Top Purity', f"{current.get('top_purity', 0):.4f}"],
            ['Optimized Top Purity', f"{optimal.get('top_purity', 0):.4f}"],
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10b981')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0fdf4'), colors.HexColor('#ffffff')]),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("Before vs After Comparison", self.section_style))
        
        curr_rr = current.get('rr', current.get('reflux_ratio', 'N/A'))
        curr_fr = current.get('fr', current.get('feed_rate', 'N/A'))
        curr_fc = current.get('fc', current.get('feed_composition', 'N/A'))
        opt_rr = optimal.get('reflux_ratio', optimal.get('rr', 'N/A'))
        opt_fr = optimal.get('feed_rate', optimal.get('fr', 'N/A'))
        opt_fc = optimal.get('feed_composition', optimal.get('fc', 'N/A'))
        
        comp_data = [
            ['Parameter', 'Current', 'AI Optimized'],
            ['Reflux Ratio', f"{curr_rr}" if isinstance(curr_rr, str) else f"{curr_rr:.2f}", 
             f"{opt_rr}" if isinstance(opt_rr, str) else f"{opt_rr:.2f}"],
            ['Feed Rate', f"{curr_fr} mol/hr" if isinstance(curr_fr, str) else f"{curr_fr:.1f} mol/hr",
             f"{opt_fr} mol/hr" if isinstance(opt_fr, str) else f"{opt_fr:.1f} mol/hr"],
            ['Feed Composition', f"{curr_fc}" if isinstance(curr_fc, str) else f"{curr_fc:.3f}",
             f"{opt_fc}" if isinstance(opt_fc, str) else f"{opt_fc:.3f}"],
            ['Reboiler Duty', f"{current.get('reboiler_duty', 0):.2f} kW", f"{optimal.get('reboiler_duty', 0):.2f} kW"],
            ['Condenser Duty', f"{current.get('condenser_duty', 0):.2f} kW", f"{optimal.get('condenser_duty', 0):.2f} kW"],
            ['Top Purity', f"{current.get('top_purity', 0):.4f}", f"{optimal.get('top_purity', 0):.4f}"],
            ['Bottom Purity', f"{current.get('bottom_purity', 0):.4f}", f"{optimal.get('bottom_purity', 0):.4f}"],
        ]
        
        comp_table = Table(comp_data, colWidths=[2*inch, 2*inch, 2*inch])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#eef2ff'), colors.HexColor('#ffffff')]),
        ]))
        story.append(comp_table)
        story.append(Spacer(1, 20))
        
        # FIXED: No <br> tags, use bullet list
        story.append(Paragraph("AI Recommendations", self.section_style))
        
        try:
            rr_change = "Increase" if float(opt_rr) > float(curr_rr) else "Decrease" if float(opt_rr) < float(curr_rr) else "Keep"
        except:
            rr_change = "Adjust"
        
        recommendations = [
            f"1. Energy Savings: Implementing the optimized parameters will save {savings:.2f} kW ({savings_pct:.1f}%) in reboiler duty.",
            f"2. Reflux Ratio: {rr_change} reflux ratio from {curr_rr} to {opt_rr}.",
            f"3. Product Quality: Top purity will change from {current.get('top_purity', 0):.4f} to {optimal.get('top_purity', 0):.4f}.",
            "4. Implementation: Gradually transition to new operating conditions while monitoring product quality."
        ]
        
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", self.bullet_style))
            story.append(Spacer(1, 4))
        
        story.append(Spacer(1, 20))
        story.append(Paragraph("Generated by DistilAI Pro - AI-Powered Optimization", self.info_style))
        
        doc.build(story, onFirstPage=self.create_header_footer, onLaterPages=self.create_header_footer)
        buffer.seek(0)
        return buffer, filename


def pdf_download_button(buffer, filename, label="Download PDF Report"):
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return st.markdown(
        f'<a href="data:application/pdf;base64,{b64}" download="{filename}" '
        f'style="text-decoration:none;">'
        f'<div style="background:linear-gradient(135deg, #667eea, #764ba2);'
        f'color:white;padding:12px 24px;border-radius:12px;'
        f'text-align:center;font-weight:600;cursor:pointer;">'
        f'📄 {label}</div></a>',
        unsafe_allow_html=True
    )