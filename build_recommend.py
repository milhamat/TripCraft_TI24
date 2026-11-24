#!/usr/bin/env python3
# Gabungkan fasilitas HOTEL + WISATA -> dataset, hitung Recommended, rebuild TF-IDF (+rec), suntik ke HTML.
# Idempoten: baris yang Description-nya sudah mengandung "Fasilitas:" dilewati (tidak dobel-enrich).
import json, re, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

CSV='data/cleaned/batam_cleaned.csv'; HTML='tripcraft_batam.html'
FAS_HOTEL='data/fasilitas_hotel.json'; FAS_WISATA='data/fasilitas_wisata.json'
df=pd.read_csv(CSV)
clean_fac=lambda s: re.sub(r'[^a-z0-9 ]',' ',str(s).lower()).strip()
def ensure(cols):
    for c in cols:
        if c not in df.columns: df[c]=pd.Series(['']*len(df),dtype='object')
def enriched(i): return 'Fasilitas:' in str(df.at[i,'Description'])

# ---- 1a. HOTEL ----
ensure(['Bintang','Fasilitas','Cocok_Untuk','Harga_Hotel'])
hot={f['nama'].strip():f for f in json.load(open(FAS_HOTEL,encoding='utf-8'))}
mh=0
for i,r in df.iterrows():
    if r['Category']!='Hotel' or enriched(i): continue
    f=hot.get(str(r['Place_Name']).strip())
    if not f: print('  ! hotel tdk match:',r['Place_Name']); continue
    mh+=1; fl=f['fasilitas']; cocok=f.get('cocok_untuk',[]); b=f.get('bintang')
    df.at[i,'Bintang']=b; df.at[i,'Fasilitas']='; '.join(fl); df.at[i,'Cocok_Untuk']='; '.join(cocok)
    df.at[i,'Harga_Hotel']=f.get('harga') or f.get('harga_estimasi_per_malam_idr','')
    df.at[i,'Description']=f"Hotel bintang {b} di {r['Area']}. Fasilitas: {', '.join(fl[:6]).lower()}. Cocok untuk {', '.join(cocok)}."
    extra=[clean_fac(x) for x in fl]+list(cocok)+[f"bintang {b}"]
    if any('restoran' in x.lower() for x in fl): extra.append('dining')
    if any('keamanan' in x.lower() or 'cctv' in x.lower() for x in fl): extra.append('aman')
    if 'budget' in cocok: extra+=['murah','hemat']
    if 'mewah' in cocok or (b and b>=5): extra.append('mewah')
    df.at[i,'Search_Text']=(str(r['Search_Text'])+' '+' '.join(extra)).lower()
print(f'  hotel ter-enrich sekarang: {mh}')

# ---- 1b. WISATA + ALAM ----
ensure(['Aktivitas','Tiket'])
ALIAS={'Palm Springs Golf & Country Club':'Palm Springs Golf Golf & Country Club'}
def key(nm):
    nm=nm.rsplit(' - ',1)[0].strip(); return ALIAS.get(nm,nm)
wis={key(f['nama']):f for f in json.load(open(FAS_WISATA,encoding='utf-8'))}
mw=0; unmatched=[]
for i,r in df.iterrows():
    if r['Category'] not in ('Wisata','Alam') or enriched(i): continue
    f=wis.get(str(r['Place_Name']).strip())
    if not f: unmatched.append(r['Place_Name']); continue
    mw+=1; fl=f.get('fasilitas',[]); akt=f.get('aktivitas',[]); cocok=f.get('cocok_untuk',[]); tk=f.get('tiket')
    df.at[i,'Fasilitas']='; '.join(fl); df.at[i,'Cocok_Untuk']='; '.join(cocok)
    df.at[i,'Aktivitas']='; '.join(akt); df.at[i,'Tiket']=tk or ''
    kind='wisata' if r['Category']=='Wisata' else 'alam'
    desc=f"Tempat {kind} di {r['Area']}, Batam."
    if fl: desc+=f" Fasilitas: {', '.join(fl[:6]).lower()}."
    if akt: desc+=f" Aktivitas: {', '.join(a.lower() for a in akt)}."
    if cocok: desc+=f" Cocok untuk {', '.join(cocok)}."
    if tk: desc+=(" Masuk gratis." if tk=='Gratis' else f" Tiket masuk Rp{tk}.")
    df.at[i,'Description']=desc
    extra=[clean_fac(x) for x in fl]+[clean_fac(a) for a in akt]+list(cocok)
    if tk=='Gratis': extra.append('gratis')
    df.at[i,'Search_Text']=(str(r['Search_Text'])+' '+' '.join(extra)).lower()
print(f'  wisata/alam ter-enrich sekarang: {mw}')
if unmatched: print('  ! wisata tdk match:',unmatched)

# ---- 2. Recommended (deterministik) ----
norm=lambda s:( (s:=s.astype(float)) - s.min())/(s.max()-s.min()+1e-9)
q=0.6*norm(df['Rating'])+0.4*norm(np.log1p(df['Jumlah_Ulasan'])); med=q.median()
df['Quality_Score']=q.round(4); df['Recommended']=(q>=med).astype(int)
print(f"  Recommended: {df['Recommended'].sum()} / Tidak: {(df['Recommended']==0).sum()}")
df.to_csv(CSV,index=False); print('  CSV tersimpan')

# ---- 3. Rebuild TF-IDF ----
pre=lambda x: re.sub(r'[^a-z0-9\s]','',str(x).lower())
vec=TfidfVectorizer(preprocessor=pre,ngram_range=(1,2),sublinear_tf=True,norm='l2')
X=vec.fit_transform(df['Search_Text']); vocab={t:int(i) for t,i in vec.vocabulary_.items()}
idf=[round(float(v),4) for v in vec.idf_]; print(f'  vocab: {len(vocab)}, matrix: {X.shape}')

# ---- 4. pois (+rec) ----
sinonim={"makan":"kuliner","jajan":"kuliner","nyemil":"kuliner","kulineran":"kuliner",
"nginep":"hotel","penginapan":"hotel","nginap":"hotel","hotel":"hotel",
"shopping":"belanja","oleholeh":"belanja","ibadah":"religi","sholat":"masjid",
"sembahyang":"religi","gereja":"religi","klenteng":"religi","pura":"religi","wihara":"vihara",
"santai":"pantai","ngopi":"kafe","kopi":"kafe","coffee":"kafe","pedes":"pedas",
"mahal":"mewah","fitness":"gym","renang":"berenang","wifi":"wifi","keluarga":"keluarga",
"camping":"berkemah","kemah":"berkemah","mendaki":"hiking","sejarah":"edukasi","gratis":"gratis"}
pois=[]; Xc=X.tocsr()
for i,r in df.iterrows():
    row=Xc.getrow(i); v=[[int(j),round(float(row[0,j]),4)] for j in row.indices]
    pois.append({"id":int(r['Place_Id']),"n":r['Place_Name'],"area":r['Area'],"c":r['Category'],
                 "lat":float(r['Lat']),"lon":float(r['Long']),"r":float(r['Rating']),
                 "ul":int(r['Jumlah_Ulasan']),"dur":int(r['Durasi_Menit']),
                 "desc":r['Description'],"rec":int(r['Recommended']),"v":v})
MODEL={"vocab":vocab,"idf":idf,"sinonim":sinonim,"pois":pois}
mj=json.dumps(MODEL,ensure_ascii=False,separators=(',',':'))

# ---- 5. Suntik HTML ----
lines=open(HTML,encoding='utf-8').read().split('\n')
for k,ln in enumerate(lines):
    if ln.startswith('const MODEL = {') or ln.startswith('const MODEL={'):
        lines[k]=f'const MODEL = {mj};'; print(f'  MODEL disuntik baris {k+1}'); break
open(HTML,'w',encoding='utf-8').write('\n'.join(lines)); print('DONE')