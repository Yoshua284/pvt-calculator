import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# KONFIGURASI HALAMAN & INJEKSI CSS KUSTOM
# =============================================================================
st.set_page_config(page_title="PVT Simulator Pro", page_icon="🛢️", layout="wide", initial_sidebar_state="expanded")

# Injeksi Custom CSS untuk mempercantik UI
st.markdown("""
    <style>
    /* Styling untuk main header */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #1E3A8A; /* Deep Blue */
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        font-weight: 400;
        color: #64748B;
        margin-bottom: 30px;
    }
    /* Styling Credit Card Author */
    .author-card {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .author-card h4 {
        margin-top: 0;
        color: #F8FAFC;
        font-size: 16px;
        border-bottom: 1px solid rgba(255,255,255,0.3);
        padding-bottom: 10px;
    }
    .author-card p {
        margin: 5px 0;
        font-size: 14px;
        font-weight: 500;
    }
    /* Styling Button */
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
        border-color: #2563EB;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

R = 10.73159

# =============================================================================
# 1. CORE ENGINE: GAS & LIQUID CORRELATIONS
# =============================================================================
class PVTEngine:
    def __init__(self, inputs):
        self.inv = inputs
        self.T_R = self.inv['Tres'] + 459.67
        self.gamma_o = 141.5 / (self.inv['API'] + 131.5)
        
        if self.inv['gas_tpc_corr'] == "Standing":
            self.Tpc = 168 + 325 * self.inv['gg'] - 12.5 * self.inv['gg']**2
            self.Ppc = 677 + 15.0 * self.inv['gg'] - 37.5 * self.inv['gg']**2
        else:
            self.Tpc = 169.2 + 349.5 * self.inv['gg'] - 74.0 * self.inv['gg']**2
            self.Ppc = 756.8 - 131.0 * self.inv['gg'] - 3.6 * self.inv['gg']**2

        yCO2, yH2S, yN2 = self.inv['yCO2']/100, self.inv['yH2S']/100, self.inv['yN2']/100
        if self.inv['gas_imp_corr'] == "Wichert-Aziz":
            A, B = yH2S + yCO2, yH2S
            eps = 120 * (A**0.9 - A**1.6) + 15 * (B**0.5 - B**4.0)
            self.Tpc_corr = self.Tpc - eps
            self.Ppc_corr = (self.Ppc * self.Tpc_corr) / (self.Tpc + B * (1 - B) * eps)
        elif self.inv['gas_imp_corr'] == "Carr-Kobayashi-Burrows":
            self.Tpc_corr = self.Tpc - 80.0 * yCO2 + 130.0 * yH2S - 250.0 * yN2
            self.Ppc_corr = self.Ppc + 440.0 * yCO2 + 600.0 * yH2S - 170.0 * yN2
        else:
            self.Tpc_corr, self.Ppc_corr = self.Tpc, self.Ppc

        self.Tpr = self.T_R / self.Tpc_corr

    def calc_z_factor(self, P):
        Ppr = P / self.Ppc_corr
        if self.inv['gas_z_corr'] == "Papay (Analitik)":
            return 1 - (3.52 * Ppr / 10**(0.9813 * self.Tpr)) + (0.274 * Ppr**2 / 10**(0.8157 * self.Tpr))
        else:
            t, y = 1.0 / self.Tpr, 0.001
            for _ in range(100):
                A = -0.06125 * Ppr * t * np.exp(-1.2 * (1 - t)**2)
                B = (y + y**2 + y**3 - y**4) / (1 - y)**3
                C = -(14.76 * t - 9.76 * t**2 + 4.58 * t**3) * y**2
                D = (90.7 * t - 242.2 * t**2 + 42.4 * t**3) * y**(2.18 + 2.82 * t)
                f = A + B + C + D
                df = (1 + 4*y + 4*y**2 - 4*y**3 + y**4) / (1 - y)**4 - 2*(14.76*t - 9.76*t**2 + 4.58*t**3)*y + (2.18 + 2.82*t)*(90.7*t - 242.2*t**2 + 42.4*t**3)*y**(1.18 + 2.82*t)
                y_new = y - f / df
                if abs(y_new - y) < 1e-7: break
                y = max(1e-6, min(y_new, 0.99))
            return max(0.05, 0.06125 * Ppr * t / y * np.exp(-1.2 * (1 - t)**2))

    def calc_Rs(self, P, corr):
        P_eff = min(P, self.inv['Pb'])
        if corr == 'Vasquez-Beggs':
            gamma_gs = self.inv['gg'] * (1 + 5.912e-5 * self.inv['API'] * 100 * np.log10(100 / 114.7))
            C1, C2, C3 = (0.0362, 1.0937, 25.7240) if self.inv['API'] <= 30 else (0.0178, 1.1870, 23.9310)
            return max(0.0, C1 * gamma_gs * P_eff**C2 * np.exp(C3 * self.inv['API'] / self.T_R))
        elif corr == 'Glaso':
            x = 2.8869 - (14.1811 - 3.3093 * np.log10(P_eff))**0.5
            return max(0.0, self.inv['gg'] * (10**x * (self.inv['API']**0.989 / self.T_R**0.172))**1.2255)
        else: # Standing Default
            x = 0.0125 * self.inv['API'] - 0.00091 * self.inv['Tres']
            return max(0.0, self.inv['gg'] * ((P_eff / 18.2 + 1.4) * 10**x)**1.2048)

    def calc_Bo(self, P, Rs, corr):
        if P > self.inv['Pb']:
            Rsb = self.calc_Rs(self.inv['Pb'], corr)
            Bob = self.calc_Bo(self.inv['Pb'], Rsb, corr)
            Co = self.calc_Co(P, Rsb)
            return Bob * np.exp(-Co * (P - self.inv['Pb']))
        
        if corr == 'Vasquez-Beggs':
            gamma_gs = self.inv['gg'] * (1 + 5.912e-5 * self.inv['API'] * 100 * np.log10(100 / 114.7))
            C1, C2, C3 = (4.677e-4, 1.751e-5, -1.811e-8) if self.inv['API'] <= 30 else (4.670e-4, 1.100e-5, 1.337e-9)
            return 1.0 + C1 * Rs + C2 * (self.inv['Tres'] - 60) * (self.inv['API'] / gamma_gs) + C3 * Rs * (self.inv['Tres'] - 60) * (self.inv['API'] / gamma_gs)
        elif corr == 'Glaso':
            F = Rs * (self.inv['gg'] / self.gamma_o)**0.526 + 0.968 * self.inv['Tres']
            A = -6.58511 + 2.91329 * np.log10(F) - 0.27683 * (np.log10(F))**2
            return 1 + 10**A
        else:
            return 0.9759 + 1.2e-4 * (Rs * (self.inv['gg'] / self.gamma_o)**0.5 + 1.25 * self.inv['Tres'])**1.2

    def calc_Co(self, P, Rs):
        return max(1e-6, (-1433 + 5*Rs + 17.2*self.inv['Tres'] - 1180*self.inv['gg'] + 12.61*self.inv['API']) / (1e5 * P))

    def calc_Cg(self, P):
        dP = 0.1
        Z1, Z2 = self.calc_z_factor(P - dP), self.calc_z_factor(P + dP)
        Z = self.calc_z_factor(P)
        return max(1e-6, (1.0 / P) - (1.0 / Z) * ((Z2 - Z1) / (2 * dP)))

    def calc_mu_o(self, Rs):
        mu_od = 10**(self.inv['Tres']**(-1.163) * np.exp(6.9824 - 0.04658 * self.inv['API'])) - 1.0
        return (10.715 * (Rs + 100)**(-0.515)) * mu_od**(5.44 * (Rs + 150)**(-0.338))
        
    def calc_mu_g(self, P, Z):
        Mg = 28.967 * self.inv['gg']
        rho_g = P * Mg / (Z * R * self.T_R)
        x = (9.379 + 0.01607 * Mg) * self.T_R**1.5 / (209.2 + 19.26 * Mg + self.T_R)
        y = 3.448 + 986.4 / self.T_R + 0.01009 * Mg
        z_ = 2.447 - 0.2224 * y
        return 1e-4 * x * np.exp(y * (rho_g / 62.4)**z_)

    def calc_brine(self, P):
        S_wt = self.inv['salinity'] / 10000.0
        Bw = 1.0 - 1.0001e-2*P + 1.33391e-4*(P**2) - 5.50654e-7*(P**3)
        mu_D = 10**(109.574 - 8.40564*np.log10(self.inv['Tres']) + 0.313314*(np.log10(self.inv['Tres']))**2 + 8.72213e-3*(np.log10(self.inv['Tres']))**3)
        mu_w = mu_D * (1 - 1.87e-3*S_wt**0.5 + 2.18e-4*S_wt**2.5 + (self.inv['Tres']**0.5 - 1.35e-2*self.inv['Tres'])* (2.76e-3*S_wt - 3.44e-4*S_wt**1.5))
        Cw = 1.0 / (7.033 * P + 0.541 * self.inv['salinity'] - 537.0 * self.inv['Tres'] + 403300.0)
        return max(1.0, Bw), mu_w, max(1e-7, Cw)

# =============================================================================
# 2. MAIN HEADER & ANTARMUKA UI STREAMLIT
# =============================================================================
st.markdown("<h1 class='main-title'>🛢️ PVT Simulator Ultimate PRO</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='sub-title'>Comprehensive 12-Parameter Reservoir Fluid Analysis Dashboard</h4>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    # --- KARTU IDENTITAS AUTHOR ---
    st.markdown("""
        <div class="author-card">
            <h4>👨‍💻 Dikembangkan Oleh:</h4>
            <p>• Rifqi Ariiq Rozaan (12224028)</p>
            <p>• Yoshua Ngasup Sitepu (12224029)</p>
            <p>• Khayru Akhdan Vradinka (12224046)</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.header("⚙️ Konfigurasi Input Data")
    
    with st.expander("📌 General Data", expanded=True):
        Tres = st.number_input("Reservoir Temp (°F):", value=180.0)
        Pi = st.number_input("Initial Pressure (psia):", value=4000.0)
    
    with st.expander("🧪 Impurities & Separator", expanded=False):
        c1, c2, c3 = st.columns(3)
        yCO2 = c1.number_input("CO2 (%):", value=2.0)
        yH2S = c2.number_input("H2S (%):", value=1.0)
        yN2 = c3.number_input("N2 (%):", value=3.0)

    with st.expander("💧 Oil Data", expanded=True):
        API = st.number_input("Oil API (°API):", value=35.0)
        Pb = st.number_input("Bubble-Point (psia):", value=2500.0)
        oil_corr = st.selectbox("Oil Correlation:", ["Standing", "Vasquez-Beggs", "Glaso"])

    with st.expander("💨 Gas & Brine Data", expanded=True):
        gg = st.number_input("Gas Gravity:", value=0.65)
        gas_tpc_corr = st.selectbox("Tpc/Ppc Corr:", ["Standing", "Sutton"])
        gas_imp_corr = st.selectbox("Impurity Corr:", ["Wichert-Aziz", "None"])
        gas_z_corr = st.selectbox("Z-factor Method:", ["Hall-Yarborough (NR-Iter)", "Papay (Analitik)"])
        salinity = st.number_input("Salinity (ppm):", value=30000.0)
        
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Calculate PVT Dashboard", use_container_width=True)

# =============================================================================
# 3. PROSES & VISUALISASI 12 KURVA
# =============================================================================
if run_btn:
    inputs = {'Tres': Tres, 'API': API, 'Pb': Pb, 'gg': gg, 
              'yCO2': yCO2, 'yH2S': yH2S, 'yN2': yN2, 'salinity': salinity,
              'gas_tpc_corr': gas_tpc_corr, 'gas_imp_corr': gas_imp_corr, 'gas_z_corr': gas_z_corr}
    
    engine = PVTEngine(inputs)
    P_range = np.linspace(100, Pi, 60)
    P_range = np.sort(np.unique(np.append(P_range, Pb)))

    rows = []
    Rsb = engine.calc_Rs(Pb, oil_corr)
    
    # Progress bar interaktif
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, P in enumerate(P_range):
        Rs = engine.calc_Rs(P, oil_corr)
        Bo = engine.calc_Bo(P, Rs, oil_corr)
        Z = engine.calc_z_factor(P)
        
        Bg_cuft = 0.02829 * Z * engine.T_R / P
        Bg = Bg_cuft * 1000 / 5.61458
        Bt = Bo + (Bg_cuft / 5.61458) * (Rsb - Rs) if P <= Pb else Bo
        
        Co = engine.calc_Co(P, Rs) if P > Pb else np.nan
        Cg = engine.calc_Cg(P)
        
        mu_o = engine.calc_mu_o(Rs)
        mu_g = engine.calc_mu_g(P, Z)
        
        rho_o = (62.4 * engine.gamma_o + 0.0764 * gg * Rs) / (5.61458 * Bo)
        rho_g = P * (28.967 * gg) / (Z * R * engine.T_R)
        
        Bw, mu_w, Cw = engine.calc_brine(P)

        rows.append({
            'P (psia)': P, 'Rs (scf/STB)': Rs, 'Bo (RB/STB)': Bo, 'Bg (RB/Mscf)': Bg, 'Bt (RB/STB)': Bt, 
            'Z-Factor': Z, 'μo (cp)': mu_o, 'μg (cp)': mu_g, 'ρo (lb/ft³)': rho_o, 'ρg (lb/ft³)': rho_g,
            'Co (psi⁻¹)': Co, 'Cg (psi⁻¹)': Cg, 'Bw (RB/STB)': Bw, 'μw (cp)': mu_w
        })
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(P_range))
        
    status_text.empty()
    progress_bar.empty()
    
    df = pd.DataFrame(rows)
    
    # Menampilkan Summary Singkat
    st.success("✅ Kalkulasi PVT Berhasil Diselesaikan!")
    
    # ---------------------------------------------------------
    # DASHBOARD 12 GRAFIK DENGAN ESTETIKA BARU
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(15, 14))
    fig.patch.set_facecolor('#F8FAFC') # Background luar grafik
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.3)
    
    plots = [
        (0, 0, 'Rs (scf/STB)', 'Solution GOR ($R_s$)', '#1E3A8A'),
        (0, 1, 'Bo (RB/STB)', 'Oil FVF ($B_o$)', '#D97706'),
        (0, 2, 'Bt (RB/STB)', 'Total FVF ($B_t$)', '#059669'),
        (1, 0, 'μo (cp)', 'Oil Viscosity ($\mu_o$)', '#4F46E5'),
        (1, 1, 'μg (cp)', 'Gas Viscosity ($\mu_g$)', '#E11D48'),
        (1, 2, 'μw (cp)', 'Water Viscosity ($\mu_w$)', '#0891B2'),
        (2, 0, 'ρo (lb/ft³)', 'Oil Density ($\\rho_o$)', '#7C3AED'),
        (2, 1, 'ρg (lb/ft³)', 'Gas Density ($\\rho_g$)', '#EA580C'),
        (2, 2, 'Z-Factor', 'Gas Z-Factor', '#65A30D'),
        (3, 0, 'Bg (RB/Mscf)', 'Gas FVF ($B_g$)', '#DB2777'),
        (3, 1, 'Co (psi⁻¹)', 'Oil Compressibility ($c_o$)', '#475569'),
        (3, 2, 'Cg (psi⁻¹)', 'Gas Compressibility ($c_g$)', '#0D9488')
    ]
    
    for r, c, key, title, color in plots:
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor('#FFFFFF') # Background dalam grafik
        
        valid_data = df.dropna(subset=[key])
        ax.plot(valid_data['P (psia)'], valid_data[key], color=color, lw=2.5)
        ax.fill_between(valid_data['P (psia)'], valid_data[key], alpha=0.1, color=color) # Tambahan efek shadow/fill
        
        ax.axvline(Pb, color='#EF4444', linestyle='--', alpha=0.8, lw=1.5, label='$P_b$')
        ax.set_title(title, fontweight='bold', fontsize=12, color='#1E293B', pad=10)
        ax.set_xlabel('Pressure (psia)', fontsize=10, color='#64748B')
        ax.tick_params(colors='#64748B', labelsize=9)
        
        # Mempercantik Grid
        ax.grid(True, linestyle='--', alpha=0.4, color='#CBD5E1')
        for spine in ax.spines.values():
            spine.set_edgecolor('#CBD5E1')
            
        if key in ['Co (psi⁻¹)', 'Cg (psi⁻¹)', 'Bg (RB/Mscf)']: 
            ax.set_yscale('log')
            
    st.pyplot(fig)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabel Data Ekspansi
    with st.expander("🗃️ Buka Tabel Data Analisis PVT Resolusi Tinggi"):
        fmt = {col: "{:.4f}" for col in df.columns}
        fmt['Co (psi⁻¹)'] = fmt['Cg (psi⁻¹)'] = "{:.3e}"
        st.dataframe(df.style.format(fmt, na_rep="-"), use_container_width=True)
else:
    # Tampilan awal (*placeholder*) saat belum di-run
    st.info("👈 Silakan atur parameter di panel samping dan klik **Calculate PVT Dashboard** untuk memulai simulasi.")