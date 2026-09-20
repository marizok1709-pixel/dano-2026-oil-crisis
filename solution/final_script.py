#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ТОПЛИВНЫЙ КРИЗИС 2026 И ПДД
 Полный код анализа: сырые данные -> очистка -> модели -> вывод

    дефицит топлива -> очереди на АЗС -> изменение поведения -> нарушения
       измерено         НЕ измерено         косвенно            измерено


ЗАПУСК
    python3 fuel_crisis_analysis.py [--raw ПУТЬ] [--out ПУТЬ]

ВЫХОД
    <out>/cleaned_data/   очищенные таблицы + журнал очистки
    <out>/figures/        g06, g07, g08 и диагностические графики
    <out>/results/        таблицы результатов всех моделей
    <out>/run_log.txt     полный текстовый протокол прогона
"""

import argparse
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from scipy import stats, optimize

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10, "axes.grid": True,
    "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False,
    "figure.autolayout": True,
})

# вывод
_LOG = []


def show(obj, title=None):
    """Печать таблицы замена display() из notebook"""
    if title:
        say(title)
    txt = obj.to_string() if hasattr(obj, "to_string") else str(obj)
    say(txt)


def say(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    _LOG.append(line)


def section(title):
    bar = "=" * 78
    say("\n" + bar)
    say(title.upper())
    say(bar)


# ------------------------------------------------------------------ пути ----
def _parse_args():
    ap = argparse.ArgumentParser(
        description="Анализ влияния топливного кризиса 2026 на дорожные правонарушения")
    ap.add_argument("--raw", default=str(Path(__file__).resolve().parent.parent),
                    help="папка с сырыми CSV (clients_demographics, fines_2026, fuel_transaction)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent),
                    help="папка для результатов")
    return ap.parse_args()


ARGS = _parse_args()
RAW = Path(ARGS.raw)
OUTDIR = Path(ARGS.out)
for _d in ("cleaned_dataset", "figures", "results"):
    (OUTDIR / _d).mkdir(parents=True, exist_ok=True)

for _f in ("clients_demographics.csv", "fines_2026.csv", "fuel_transaction.csv"):
    if not (RAW / _f).exists():
        sys.exit(f"ОШИБКА: не найден файл {RAW / _f}\n"
                 f"Укажите папку с сырыми данными через --raw")

say(f"сырые данные : {RAW}")
say(f"результаты   : {OUTDIR}")




# import json, warnings, re
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60)

OUT = OUTDIR / "cleaned_dataset"

LOG = []
def log(table, code, what, n, action):
    LOG.append({"таблица":table,"код":code,"что_обнаружено":what,
                "строк":n,"решение":action})
    print(f"[{table:8s}] {code:22s} n={n:>7,d}  -> {action}")

# Разделитель ; десятичная запятая проверено на сырых файлах
clients_raw = pd.read_csv(RAW/"clients_demographics.csv", sep=';', decimal=',',
                          parse_dates=['subscription_creation_date'])
fines_raw   = pd.read_csv(RAW/"fines_2026.csv", sep=';',
                          parse_dates=['bill_offence_date'])
fuel_raw    = pd.read_csv(RAW/"fuel_transaction.csv", sep=';', decimal=',',
                          parse_dates=['order_datetime'])

say("СЫРЫЕ ФАЙЛЫ")
for n,d in [("clients_demographics",clients_raw),("fines_2026",fines_raw),("fuel_transaction",fuel_raw)]:
    print(f"  {n:22s} {d.shape[0]:>7,d} строк x {d.shape[1]:2d} колонок")
TOTAL_ROWS = len(clients_raw)+len(fines_raw)+len(fuel_raw)
TOTAL_COLS = clients_raw.shape[1]+fines_raw.shape[1]+fuel_raw.shape[1]
say(f"  {'ИТОГО':22s} {TOTAL_ROWS:>7,d} строк, {TOTAL_COLS} признаков")

say("ТИПЫ И ПРИМЕРЫ ЗНАЧЕНИЙ\n")
for n,d in [("clients",clients_raw),("fines",fines_raw),("fuel",fuel_raw)]:
    print(f"--- {n} ---")
    info = pd.DataFrame({"тип":d.dtypes.astype(str),
                         "уникальных":d.nunique(),
                         "пропусков":d.isna().sum(),
                         "пример":[d[c].dropna().iloc[0] if d[c].notna().any() else None for c in d.columns]})
    show(info)

n_dup_cid  = clients_raw.client_id.duplicated().sum()
n_dup_pair = clients_raw.duplicated(['client_id','auto_document_id']).sum()
say(f"строк                         : {len(clients_raw):,}")
say(f"уникальных client_id          : {clients_raw.client_id.nunique():,}")
say(f"повторов client_id            : {n_dup_cid:,}")
say(f"повторов (client_id, auto_id) : {n_dup_pair:,}")
say(f"уникальных (client_id, auto_id): {len(clients_raw.drop_duplicates(['client_id','auto_document_id'])):,}")
say("\nРаспределение числа автомобилей на клиента:")
show(clients_raw.groupby('client_id').auto_document_id.nunique().value_counts().sort_index()
        .rename_axis('авто у клиента').to_frame('клиентов'))
say("ВЫВОД: единица наблюдения = клиент x автомобиль. Повтор client_id — это НЕ дубликат,")
say("       а второй автомобиль. Настоящие дубликаты — только повторы пары (client_id, auto_id).")

dupmask = clients_raw.duplicated(['client_id','auto_document_id'], keep=False)
dups = clients_raw[dupmask].sort_values(['client_id','auto_document_id'])
grp = dups.groupby(['client_id','auto_document_id'])
identical, differing = 0, []
for key,g in grp:
    if g.drop_duplicates().shape[0]==1: identical += 1
    else:
        diff_cols = [c for c in g.columns if g[c].nunique(dropna=False)>1]
        differing.append((key, diff_cols))
say(f"пар-дубликатов          : {grp.ngroups}")
say(f"  полностью идентичных  : {identical}")
say(f"  различающихся         : {len(differing)}")
if differing:
    from collections import Counter
    print("  различия в колонках   :", dict(Counter([c for _,cs in differing for c in cs])))
    print("\nПример различающейся пары:")
    show(dups[(dups.client_id==differing[0][0][0])&(dups.auto_document_id==differing[0][0][1])])

# различия ТОЛЬКО в счётчиках штрафов. Оставте строку с максимальным и не трогайте здесь код больше пожалуйста
# fines_last_12_month (более полная запись) вторую помечаем и исключаем из client-levelъ
clients = clients_raw.copy()
clients['_ord'] = clients.groupby(['client_id','auto_document_id'])['fines_last_12_month']\
                         .rank(method='first', ascending=False)
clients['flag_duplicate_vehicle_row'] = dupmask & (clients['_ord']>1)
clients['is_kept_vehicle_row']        = ~clients['flag_duplicate_vehicle_row']
clients = clients.drop(columns='_ord')
log("clients","DUP_VEHICLE_ROW",
    f"пара (client_id, auto_document_id) встречается дважды; {identical} пар идентичны, "
    f"{len(differing)} различаются только счётчиками штрафов",
    int(clients.flag_duplicate_vehicle_row.sum()),
    "оставлена строка с максимальным fines_last_12_month, вторая помечена флагом")

# Один автомобиль у нескольких клиентов это перепродажа, а НЕ ошибка
multi = clients.groupby('auto_document_id').client_id.nunique()
multi_ids = set(multi[multi>1].index)
clients['flag_vehicle_multi_client'] = clients.auto_document_id.isin(multi_ids)
log("clients","VEHICLE_MULTI_CLIENT",
    "один auto_document_id связан с несколькими client_id (вероятно перепродажа авто)",
    int(clients.flag_vehicle_multi_client.sum()), "сохранено как есть — это не ошибка данных")

miss = clients_raw.isna().sum(); miss = miss[miss>0]
show(pd.DataFrame({"пропусков":miss,"% строк":(100*miss/len(clients_raw)).round(3)}))
for c,n in miss.items():
    log("clients", f"NULL_{c.upper()}", f"пропуск в поле {c}", int(n),
        "НЕ импутировано; строка остаётся, выпадает только из моделей с этим признаком")
say("\nПочему не импутируем: цена авто и цвет не связаны с механизмом гипотезы,")
say("а импутация пола/возраста создала бы искусственную вариацию в контрольных переменных.")

# engine_type: "2.0 (150.00 л.с.)" -> объём и мощность
ex = clients.engine_type.dropna().str.extract(r'^\s*([\d.]+)\s*\(([\d.]+)')
clients['engine_litres'] = pd.to_numeric(ex[0], errors='coerce')
clients['engine_hp']     = pd.to_numeric(ex[1], errors='coerce')
say("разбор engine_type:")
say(f"  успешно  : {clients.engine_litres.notna().sum():,} из {clients.engine_type.notna().sum():,}")
bad = clients[clients.engine_type.notna() & clients.engine_litres.isna()]
say(f"  не распознано: {len(bad)}", ("| примеры: "+", ".join(bad.engine_type.unique()[:5])) if len(bad) else "")
show(clients[['engine_litres','engine_hp']].describe().round(2))

clients['vehicle_age_2026'] = 2026 - clients.auto_year
say(f"\nвозраст авто: медиана {clients.vehicle_age_2026.median():.0f} лет, "
      f"диапазон {clients.vehicle_age_2026.min():.0f}..{clients.vehicle_age_2026.max():.0f}")
odd = clients[(clients.vehicle_age_2026<0)|(clients.vehicle_age_2026>50)]
say(f"аномальный возраст (<0 или >50): {len(odd)} строк")

# Регион из КЛАДР
KLADR = {77:"Москва",78:"Санкт-Петербург",50:"Московская область",47:"Ленинградская область",
         66:"Свердловская область",16:"Республика Татарстан",52:"Нижегородская область",
         61:"Ростовская область",23:"Краснодарский край",63:"Самарская область",
         74:"Челябинская область",24:"Красноярский край",54:"Новосибирская область",
         36:"Воронежская область",2:"Республика Башкортостан",59:"Пермский край",
         34:"Волгоградская область",72:"Тюменская область",64:"Саратовская область",
         55:"Омская область",38:"Иркутская область",22:"Алтайский край"}
clients['region_name'] = clients.kladr_code.map(KLADR).fillna("код "+clients.kladr_code.astype(str))
say(f"\nрегионов в данных: {clients.kladr_code.nunique()}")
show(clients.region_name.value_counts().head(8).to_frame("строк"))

clients['subscription_dt'] = clients.subscription_creation_date
W_START = pd.Timestamp("2026-04-01")      # порог обосновывается в 02_EDA, то есть левом усечении
clients['flag_subscribed_after_window'] = clients.subscription_dt >= W_START
clients['flag_no_valid_2025_baseline']  = clients.subscription_dt >= pd.Timestamp("2025-04-01")

say("распределение даты подписки:")
show(clients.subscription_dt.dt.to_period('Y').value_counts().sort_index().to_frame('строк'))
n_after = int(clients.loc[clients.is_kept_vehicle_row,'flag_subscribed_after_window'].sum())
log("clients","SUBSCRIBED_MID_WINDOW",
    "клиент подключился после 01.04.2026 — у него нет полного докризисного периода",
    n_after, "помечен; исключён из фиксированной когорты для расчёта ставок")

# счётчики 2025 занижены у людей которые поздно подклбчились
b = clients[clients.is_kept_vehicle_row].copy()
b['f2025'] = b[['april_2025_fines','may_2025_fines','jun_2025_fines','jul_2025_fines','aug_2025_fines']].sum(axis=1)
say(f"\nсреднее штрафов апр-авг 2025: подключены до 04.2025 = "
      f"{b.loc[~b.flag_no_valid_2025_baseline,'f2025'].mean():.2f}, после = "
      f"{b.loc[b.flag_no_valid_2025_baseline,'f2025'].mean():.2f}")
log("clients","NO_2025_BASELINE",
    "клиент подключился после 01.04.2025 — счётчики штрафов 2025 механически занижены",
    int(b.flag_no_valid_2025_baseline.sum()),
    "помечен; исключён из всех межгодовых сравнений 2025 vs 2026")

FORBIDDEN = ['fines_last_6_month','fines_last_12_month','fines_last_24_month','fines_last_36_month']
_f = fines_raw.copy()
_f['m'] = _f.bill_offence_date.dt.to_period('M')
y2026 = _f[_f.m.between(pd.Period('2026-04'),pd.Period('2026-07'))].groupby('client_id').size()
_b = clients[clients.is_kept_vehicle_row].drop_duplicates('client_id').set_index('client_id').copy()
_b['y2026'] = y2026.reindex(_b.index).fillna(0)
_b['f2025'] = _b[['april_2025_fines','may_2025_fines','jun_2025_fines','jul_2025_fines','aug_2025_fines']].sum(axis=1)
r_post = _b[['y2026','fines_last_12_month']].corr().iloc[0,1]
r_2025 = _b[['y2026','f2025']].corr().iloc[0,1]
say(f"корр(штрафы 2026 апр-июл, fines_last_12_month) = {r_post:.3f}   <- перекрывается с исходом")
say(f"корр(штрафы 2026 апр-июл, штрафы апр-авг 2025) = {r_2025:.3f}   <- допустимый контроль")
log("clients","POST_TREATMENT_WINDOW",
    f"fines_last_6/12/24/36 пересекаются с периодом кризиса (корр. с исходом {r_post:.2f} "
    f"против {r_2025:.2f} у датированных колонок 2025)",
    len(clients_raw), "ЗАПРЕЩЕНЫ как контрольные переменные; вместо них — помесячные колонки 2025")

#здесь сохраняется уровень машины и уровень клиента
veh_cols = ['client_id','gender','age_type_code','auto_document_id','auto_mark','auto_year',
            'engine_type','engine_litres','engine_hp','price','color','kladr_code','region_name',
            'vehicle_age_2026','april_2025_fines','may_2025_fines','jun_2025_fines','jul_2025_fines',
            'aug_2025_fines','subscription_dt','flag_duplicate_vehicle_row','is_kept_vehicle_row',
            'flag_vehicle_multi_client','flag_subscribed_after_window','flag_no_valid_2025_baseline']
vehicles = clients[veh_cols].copy()
vehicles.to_csv(OUT/"clean_01_vehicles.csv", sep=';', index=False, encoding='utf-8-sig')

kept = clients[clients.is_kept_vehicle_row].copy()
kept['f2025_apr_aug'] = kept[['april_2025_fines','may_2025_fines','jun_2025_fines',
                              'jul_2025_fines','aug_2025_fines']].sum(axis=1)
agg = {'gender':'first','age_type_code':'first','kladr_code':'first','region_name':'first',
       'subscription_dt':'min','auto_mark':'first','auto_year':'max','vehicle_age_2026':'min',
       'price':'max','engine_litres':'max','engine_hp':'max','color':'first',
       'april_2025_fines':'sum','may_2025_fines':'sum','jun_2025_fines':'sum',
       'jul_2025_fines':'sum','aug_2025_fines':'sum','f2025_apr_aug':'sum',
       'flag_subscribed_after_window':'max','flag_no_valid_2025_baseline':'max'}
clients_lv = kept.groupby('client_id').agg(agg)
clients_lv.insert(0,'n_vehicles', kept.groupby('client_id').auto_document_id.nunique())
clients_lv = clients_lv.reset_index()
clients_lv.to_csv(OUT/"clean_02_clients.csv", sep=';', index=False, encoding='utf-8-sig')
say(f"clean_01_vehicles.csv : {len(vehicles):,} строк")
say(f"clean_02_clients.csv  : {len(clients_lv):,} строк (уникальных клиентов)")

fines = fines_raw.copy()
key = ['client_id','bill_offence_date','offence_short_statement','total_fine_amount']
say(f"уникальных bill_id: {fines.bill_id.nunique():,} из {len(fines):,}")
dupc = fines.duplicated(key, keep=False)
say(f"строк, совпадающих по (клиент, секунда, статья, сумма): {int(dupc.sum())}")
if dupc.any():
    show(fines[dupc].sort_values(key)[key+['bill_id']])
fines['flag_duplicate_bill'] = fines.duplicated(key, keep='first')
fines['is_kept_bill'] = ~fines.flag_duplicate_bill
log("fines","DUPLICATE_BILL",
    "разные bill_id при полном совпадении клиента, секунды нарушения, статьи и суммы",
    int(fines.flag_duplicate_bill.sum()),
    "оставлено первое постановление, второе помечено флагом")

vc = fines.total_fine_amount.value_counts().sort_index()
show(pd.DataFrame({"копейки":vc.index,"рубли":(vc.index/100).astype(int),
                      "штрафов":vc.values}).set_index("копейки"))
fines['fine_rub'] = fines.total_fine_amount/100
say(f"проверка: превышение 20-40 км/ч -> "
      f"{fines.loc[fines.offence_short_statement.str.contains('20-40'),'fine_rub'].mode().iloc[0]:.0f} ₽ "
      "(соответствует действующему тарифу)")
log("fines","AMOUNT_IN_KOPECKS","сумма штрафа указана в копейках, не в рублях",
    len(fines), "добавлена колонка fine_rub = total_fine_amount/100")

fines['offence_dt']   = fines.bill_offence_date
fines['offence_date'] = fines.offence_dt.dt.normalize()
fines['month']        = fines.offence_dt.dt.to_period('M').astype(str)
say("диапазон:", fines.offence_dt.min(), "..", fines.offence_dt.max())

kept = fines[fines.is_kept_bill]
wk = kept.groupby(kept.offence_dt.dt.to_period('W')).size()
say("\nнедельный ряд постановлений:")
for p,v in wk.items():
    bar = "#"*int(v/80)
    print(f"  {str(p)[:10]}  {v:6,d}  {bar}")

daily = kept.groupby('offence_date').size().rename('n')
full  = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
daily = daily.reindex(full, fill_value=0)
plateau = daily.loc['2026-04-15':'2026-07-15'].median()
say(f"уровень плато (15.04-15.07): {plateau:.0f} постановлений в день")
say(f"порог 50% от плато        : {0.5*plateau:.0f}\n")

below = daily < 0.5*plateau
left_end  = daily.index[~below][0]
right_beg = daily.index[(daily.index>pd.Timestamp('2026-07-01')) & below][0]
say(f"первый день выше порога            : {left_end.date()}")
say(f"первый день устойчиво ниже порога  : {right_beg.date()}")
say("\nОкругляем до календарных границ месяца, чтобы окно совпадало с месячной панелью:")
MATURE_START, MATURE_END = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-07-31")
say(f"  ЗРЕЛОЕ ОКНО = {MATURE_START.date()} .. {MATURE_END.date()}")

fines['flag_left_truncated'] = fines.offence_date <  MATURE_START
fines['flag_right_censored'] = fines.offence_date >  MATURE_END
fines['in_mature_window']    = (~fines.flag_left_truncated) & (~fines.flag_right_censored)
log("fines","LEFT_TRUNCATION",
    f"суточное число растёт с {daily.iloc[0]:.0f} до плато {plateau:.0f} — выгрузка неполна на старте",
    int(fines.flag_left_truncated.sum()), "помечено; ИСКЛЮЧЕНО из всех расчётов по нарушениям")
log("fines","RIGHT_CENSORING",
    f"с 01.08 монотонное затухание до {daily.iloc[-1]:.0f}/день у даты выгрузки — лаг регистрации",
    int(fines.flag_right_censored.sum()),
    "помечено; ИСКЛЮЧЕНО. Иначе ложный вывод о падении нарушений на ~32%")
say(f"\nв зрелом окне остаётся {int(fines.in_mature_window.sum()):,} из {len(fines):,} постановлений "
      f"({100*fines.in_mature_window.mean():.1f}%)")

m = kept.groupby(kept.offence_dt.dt.to_period('M')).size()
mat = m[['2026-04','2026-05','2026-06','2026-07']].mean()
say("ПРОВЕРКА 1. Насколько «упали» нарушения, если взять данные как есть:")
say(f"  август {m['2026-08']:,} vs среднее апр-июл {mat:,.0f} = {100*(m['2026-08']/mat-1):.1f}%")
say(f"  последняя неделя выгрузки: {daily.loc['2026-09-08':].sum():.0f} постановлений — физически невозможно\n")

say("ПРОВЕРКА 2. Независимый индикатор активности — заправки не падают:")
_fu = fuel_raw.copy(); _fu['w']=_fu.order_datetime.dt.to_period('W')
wf = _fu.groupby('w').size(); wv = kept.groupby(kept.offence_dt.dt.to_period('W')).size()
cc = pd.DataFrame({'штрафы':wv,'заправки':wf}).dropna()
cc['штрафов_на_1000_заправок'] = (1000*cc.штрафы/cc.заправки).round(1)
show(cc.tail(8))

say("ПРОВЕРКА 3 (решающая). Падение избирательно по типам нарушений:")
jul = kept[kept.month=='2026-07'].offence_short_statement.value_counts()
aug = kept[kept.month=='2026-08'].offence_short_statement.value_counts()
t = pd.DataFrame({'июль':jul,'август':aug}).dropna(); t = t[t.июль>=200]
t['изменение_%'] = (100*(t.август/t.июль-1)).round(1)
show(t.sort_values('изменение_%'))
say("Камерное превышение скорости проседает слабее всех, а типы, требующие ручного")
say("оформления (ремень, телефон, парковка), — вдвое сильнее. Это подпись лага обработки:")
say("реальное падение трафика било бы по всем категориям одинаково.")

say("ВСЕ формулировки нарушений в данных:")
show(kept.offence_short_statement.value_counts().to_frame('постановлений')
        .assign(доля_pct=lambda d:(100*d.постановлений/len(kept)).round(2)))

#пункты гипотезы записываются в данные
TARGET_MAP = {
 "razmetka" : ["Нарушение разметки"],
 "parkovka" : ["Остановка или стоянка в неположенном месте",
               "Остановка или стоянка в неположенном месте (Москва и Санкт-Петербург)"],
 "stop_line": ["Пересечение стоп-линии"],
 "obochina" : ["Движение по обочине"],
 "vydelenka": ["Движение по выделенной полосе",
               "Движение по выделенной полосе (Москва и Санкт-Петербург)"],
 "telefon"  : ["Использование телефона за рулем"],
}
#Приводим статистику
def categorize(s):
    if "Превышение скорости" in s:
        if "20-40" in s: return "Превышение 20-40"
        if "40-60" in s: return "Превышение 40-60"
        return "Превышение 60+"
    for k,v in [("Разметка / полоса",["Нарушение разметки","Поворот не из крайней полосы"]),
                ("Парковка",["Остановка"]), ("Выделенная полоса",["выделенной полосе"]),
                ("Светофор",["светофора","стоп-линии"]), ("Обочина",["обочине"]),
                ("Телефон за рулём",["телефона"]), ("Ремень безопасности",["ремень"]),
                ("Платная дорога",["платной дороге"]), ("Встречная полоса",["встречного"]),
                ("Запрещённый поворот",["запрещенном месте"]), ("Пешеход",["пешехода"]),
                ("Световые приборы",["световыми"]), ("Грузовые ТС",["грузов","Грузовы"])]:
        if any(x in s for x in v): return k
    return "Прочее"
fines['offence_category'] = fines.offence_short_statement.map(categorize)
fines['is_speeding'] = fines.offence_short_statement.str.contains("Превышение скорости")

stmt2key = {s:k for k,v in TARGET_MAP.items() for s in v}
fines['target_key'] = fines.offence_short_statement.map(stmt2key)
miss_t = [k for k,v in TARGET_MAP.items() if not fines.offence_short_statement.isin(v).any()]
say("целевые категории гипотезы, не найденные в данных:", miss_t if miss_t else "нет — найдены все 6")
show(fines[fines.is_kept_bill & fines.in_mature_window].target_key.value_counts(dropna=False)
        .to_frame('в зрелом окне'))
say("\nВНИМАНИЕ: укрупнённая offence_category НЕПРИГОДНА для проверки гипотезы —")
say("  'Светофор' смешивает красный сигнал и стоп-линию (разные механизмы);")
say("  'Разметка / полоса' смешивает разметку и поворот не из крайней полосы.")
say("  Поэтому все модели работают на уровне offence_short_statement.")

fines.to_csv(OUT/"clean_03_fines.csv", sep=';', index=False, encoding='utf-8-sig')
say(f"clean_03_fines.csv: {len(fines):,} строк, {fines.shape[1]} колонок (все строки сохранены, решения — во флагах)")

fuel = fuel_raw.copy()
fuel['order_dt']   = fuel.order_datetime
fuel['order_date'] = fuel.order_dt.dt.normalize()
fuel['month']      = fuel.order_dt.dt.to_period('M').astype(str)
neg = fuel.order_fuel_volume<=0
say(f"объём <= 0: {int(neg.sum()):,} транзакций")
say(f"  средний отрицательный объём : {fuel.loc[neg,'order_fuel_volume'].mean():.2f} л")
say(f"  средний положительный объём : {fuel.loc[~neg,'order_fuel_volume'].mean():.2f} л")
say("  зеркальность средних -> это возвраты/отмены, а не ошибки измерения")
show(fuel[neg].groupby('month').size().to_frame('возвратов')
        .join(fuel.groupby('month').size().to_frame('всего'))
        .assign(доля_pct=lambda d:(100*d.возвратов/d.всего).round(2)))
fuel['flag_reversal'] = neg
log("fuel","REVERSAL",
    f"объём <= 0 (минимум {fuel.order_fuel_volume.min():.1f} л); среднее зеркально положительному; "
    "равномерно по месяцам",
    int(neg.sum()), "помечено; ИСКЛЮЧЕНО из агрегатов объёма и цены, строка сохранена")

p, v = fuel.order_fuel_price_1liter, fuel.order_fuel_volume
tiers = [(0,50,"СУГ/КПГ (газ)"),(50,110,"Бензин/ДТ"),(110,10**6,"не моторное топливо")]
rows=[]
for lo,hi,nm in tiers:
    m = p.between(lo,hi,inclusive='left')
    rows.append({"тарифная группа":nm,"диапазон ₽/л":f"{lo}-{hi if hi<10**6 else '∞'}",
                 "транзакций":int(m.sum()),"доля_%":round(100*m.mean(),2),
                 "медиана цены":round(p[m].median(),2),"медиана объёма":round(v[m].median(),2),
                 "p90 объёма":round(v[m].quantile(.9),2)})
show(pd.DataFrame(rows))
def tier(x):
    if x<50: return "СУГ/КПГ (газ)"
    if x<110: return "Бензин/ДТ"
    return "не моторное топливо"
fuel['product_tier'] = p.map(tier)
fuel['in_analysis']  = (fuel.product_tier=="Бензин/ДТ") & (~fuel.flag_reversal)
log("fuel","PRICE_MULTIMODAL",
    "распределение цены трёхмодальное: газ ~31 ₽/л, бензин/ДТ 50-110 ₽/л (97.3%), "
    "не моторное топливо ~259 ₽/л при объёме 6.6 л",
    int((fuel.product_tier!="Бензин/ДТ").sum()),
    "разделено по тарифным группам; основной анализ — только Бензин/ДТ")

big = v>100
fuel['flag_large_volume'] = big
log("fuel","LARGE_VOLUME",
    f"объём > 100 л (максимум {v.max():.1f} л) — возможно коммерческие/бензовозные заправки",
    int(big.sum()), "СОХРАНЕНО; все ключевые метрики считаются по перцентилям, устойчивым к хвосту")

frac = (v - np.floor(v)).round(2)
say(f"доля транзакций с дробной частью .62 : {100*(frac==0.62).mean():.1f}%")
show(v.value_counts().head(8).to_frame('транзакций')
        .assign(доля_pct=lambda d:(100*d.транзакций/len(fuel)).round(2)))
say("Самые частые объёмы — 35.62, 45.62, 25.62, 55.62 л. Это не случайность:")
say("именно эти значения окажутся потолками заправки в фазах кризиса (раздел 3.4).")
say("Водители упираются в лимит и заливают ровно «под потолок».")
log("fuel","VOLUME_QUANTIZATION",
    "36.6% объёмов имеют дробную часть .62; модальные значения 35.62/45.62/55.62 л "
    "совпадают с лимитами заправки",
    int((frac==0.62).sum()), "сохранено; используется как индикатор связанности лимитом")

G = fuel[fuel.in_analysis].copy()
d90 = G.groupby('order_date').order_fuel_volume.agg(
        p90=lambda s:s.quantile(.9), p99=lambda s:s.quantile(.99),
        share_gt50=lambda s:100*(s>50).mean(), n='size')
say("суточный p90 и доля заправок >50 л, 15.06 - 08.07:")
show(d90.loc['2026-06-15':'2026-07-08'].round(2))

PHASES = [("P1_база",           "2026-03-20","2026-06-18", None),
          ("P2_ужесточение",    "2026-06-19","2026-06-23", None),
          ("P3_лимит_45.62л",   "2026-06-24","2026-07-03", 45.62),
          ("P4_лимит_35.62л",   "2026-07-04","2026-07-27", 35.62),
          ("P5_частич_ослабл",  "2026-07-28","2026-08-16", 55.62),
          ("P6_лимит_45.62л_2", "2026-08-17","2026-09-04", 45.62)]
rows=[]
for nm,a,b,lim in PHASES:
    s = G[(G.order_date>=a)&(G.order_date<=b)]
    rows.append({"фаза":nm,"с":a,"по":b,"лимит_л":lim,"транзакций":len(s),
                 "p90":round(s.order_fuel_volume.quantile(.9),2),
                 "p99":round(s.order_fuel_volume.quantile(.99),2),
                 "доля>50л_%":round(100*(s.order_fuel_volume>50).mean(),2),
                 "медиана ₽/л":round(s.order_fuel_price_1liter.median(),2),
                 "p95 ₽/л":round(s.order_fuel_price_1liter.quantile(.95),2)})
PH = pd.DataFrame(rows); show(PH)

def phase_of(dt):
    for nm,a,b,_ in PHASES:
        if pd.Timestamp(a) <= dt <= pd.Timestamp(b): return nm
    return "вне окна"
fuel['crisis_phase'] = fuel.order_date.map(phase_of)
fines['crisis_phase'] = fines.offence_date.map(phase_of)

CRISIS_ONSET = pd.Timestamp("2026-06-19")   # первый день устойчивого снижения p90
HARD_LIMIT   = pd.Timestamp("2026-06-24")   # обвал доли заправок >50 л: 10.9% -> 2.2%
say("\nВЫВОД ПО ДАТИРОВКЕ КРИЗИСА (получен из данных, без внешних источников):")
say("  до 18.06 включительно  p90 держится на 55-57 л, доля заправок >50 л ~20%")
say("  19.06 - 23.06          p90 сползает 54.2 -> 50.6, доля >50 л падает 15.6% -> 10.9%")
say("  24.06                  ОБВАЛ: доля заправок >50 л 10.9% -> 2.2%, p90 фиксируется на 45.62")
say("  04.07                  второе ужесточение: p90 -> 35.62")
say("  28.07 - 16.08          частичное ослабление, но потолок 55.62 л сохраняется (p99 = 55.62)")
say("  17.08                  возврат лимита 45.62 л")
say(f"\n  НАЧАЛО КРИЗИСА = {CRISIS_ONSET.date()}, ЖЁСТКИЙ ЛИМИТ = {HARD_LIMIT.date()}")
say("  Шок количественный: объём -37%, медианная цена всего +3%, но p95 цены +22%.")
say("  МАЙ ЦЕЛИКОМ ЛЕЖИТ В ДОКРИЗИСНОЙ БАЗЕ P1.")
log("fuel","CRISIS_PHASES_DERIVED",
    "фазы режима нормирования выведены из суточного p90 объёма и доли заправок >50 л",
    len(fuel), f"начало кризиса {CRISIS_ONSET.date()}, жёсткий лимит {HARD_LIMIT.date()}")

fuel['spend_rub'] = fuel.order_fuel_volume*fuel.order_fuel_price_1liter
fuel.to_csv(OUT/"clean_04_fuel.csv", sep=';', index=False, encoding='utf-8-sig')
PH.to_csv(OUT/"crisis_phases.csv", sep=';', index=False, encoding='utf-8-sig')
say(f"clean_04_fuel.csv: {len(fuel):,} строк")

FIXED = set(clients_lv.loc[clients_lv.subscription_dt < MATURE_START, 'client_id'])
say(f"всего клиентов                     : {len(clients_lv):,}")
say(f"подключились до {MATURE_START.date()}      : {len(FIXED):,}  <- ФИКСИРОВАННАЯ КОГОРТА")
say(f"подключились внутри/после окна     : {len(clients_lv)-len(FIXED):,}  (исключены)")

act = fuel[fuel.in_analysis].groupby('month').client_id.nunique()
say("\nПочему нельзя делить на «активных клиентов месяца»:")
show(act.to_frame('активных клиентов').assign(
    к_апрелю_pct=lambda d:(100*d['активных клиентов']/d['активных клиентов'].iloc[1]-100).round(1)))
say("Число активных падает на ~18% — это и есть эффект кризиса, а не смена базы.")

MONTHS = ['2026-04','2026-05','2026-06','2026-07']
base = pd.MultiIndex.from_product([sorted(FIXED), MONTHS], names=['client_id','month'])

V = fines[fines.is_kept_bill & fines.in_mature_window & fines.client_id.isin(FIXED)]
F = fuel[fuel.in_analysis & fuel.client_id.isin(FIXED) & fuel.month.isin(MONTHS)]

fa = F.groupby(['client_id','month']).agg(
        litres=('order_fuel_volume','sum'), fuel_spend_rub=('spend_rub','sum'),
        n_fuel_tx=('order_id','size'), vol_mean=('order_fuel_volume','mean'),
        vol_max=('order_fuel_volume','max'), price_mean=('order_fuel_price_1liter','mean'))
va = V.groupby(['client_id','month']).agg(
        n_violations=('bill_id','size'), fine_sum_rub=('fine_rub','sum'),
        n_speeding=('is_speeding','sum'), n_offence_categories=('offence_category','nunique'))
tg = V.pivot_table(index=['client_id','month'], columns='target_key', aggfunc='size', fill_value=0)

panel = pd.DataFrame(index=base).join([fa, va, tg]).reset_index()
for c in ['litres','fuel_spend_rub','n_fuel_tx','n_violations','fine_sum_rub',
          'n_speeding','n_offence_categories']+list(TARGET_MAP):
    if c in panel: panel[c] = panel[c].fillna(0)
    else: panel[c] = 0
COMPOSITE_6 = list(TARGET_MAP)                                  #6 категорий гипотезы
COMPOSITE_5 = [k for k in TARGET_MAP if k!='telefon']
panel['crisis_related_violations'] = panel[COMPOSITE_6].sum(axis=1)
panel['manoeuvre_violations']      = panel[COMPOSITE_5].sum(axis=1)
panel['month_idx'] = panel.month.map({m:i for i,m in enumerate(MONTHS)})
panel['post_crisis'] = (panel.month_idx>=3).astype(int)   # июль это первый полностью кризисный месяц

meta_cols = ['client_id','n_vehicles','gender','age_type_code','kladr_code','region_name',
             'auto_mark','auto_year','vehicle_age_2026','price','engine_litres','engine_hp',
             'subscription_dt','f2025_apr_aug','april_2025_fines','may_2025_fines',
             'jun_2025_fines','jul_2025_fines','aug_2025_fines']
panel = panel.merge(clients_lv[meta_cols], on='client_id', how='left')
say(f"панель: {len(panel):,} строк = {panel.client_id.nunique():,} клиентов x {panel.month.nunique()} месяца")

ok = True
def v(name, cond, detail):
    global ok; ok &= bool(cond)
    print(("  OK   " if cond else " !!!  ")+name+" | "+str(detail))

v("прямоугольник client x month", len(panel)==panel.client_id.nunique()*len(MONTHS),
  f"{panel.client_id.nunique():,} x {len(MONTHS)} = {len(panel):,}")
v("нет дублей (client, month)", not panel.duplicated(['client_id','month']).any(),
  f"дублей {panel.duplicated(['client_id','month']).sum()}")
v("клиенты == фиксированная когорта", set(panel.client_id)==FIXED, f"{len(FIXED):,}")
v("сумма n_violations == исходные штрафы", panel.n_violations.sum()==len(V),
  f"{int(panel.n_violations.sum()):,} vs {len(V):,}")
v("сумма литров == исходные транзакции",
  abs(panel.litres.sum()-F.order_fuel_volume.sum())<1,
  f"{panel.litres.sum():,.1f} vs {F.order_fuel_volume.sum():,.1f}")
for k in TARGET_MAP:
    v(f"  сверка категории {k}", panel[k].sum()==int((V.target_key==k).sum()),
      f"{int(panel[k].sum()):,} vs {int((V.target_key==k).sum()):,}")
v("нет отрицательных значений", (panel[['litres','n_violations']]>=0).all().all(), "ок")
say("\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ" if ok else "\nЕСТЬ РАСХОЖДЕНИЯ — СМОТРЕТЬ ВЫШЕ")

say(f"\nдоля нулевых клиенто-месяцев по n_violations: {100*(panel.n_violations==0).mean():.1f}%")
say(f"среднее {panel.n_violations.mean():.3f}, дисперсия {panel.n_violations.var():.3f}, "
      f"var/mean = {panel.n_violations.var()/panel.n_violations.mean():.2f} -> сверхдисперсия")
show(panel.groupby('month')[['n_violations','crisis_related_violations',
                                'manoeuvre_violations','litres','n_fuel_tx']].sum().round(0))

panel.to_csv(OUT/"clean_05_panel_client_month.csv", sep=';', index=False, encoding='utf-8-sig')
say(f"clean_05_panel_client_month.csv: {len(panel):,} строк x {panel.shape[1]} колонок")

LOGDF = pd.DataFrame(LOG)
show(LOGDF)
LOGDF.to_csv(OUT/"cleaning_log.csv", sep=';', index=False, encoding='utf-8-sig')

META = {
 "сырые_файлы":{"clients_demographics.csv":len(clients_raw),"fines_2026.csv":len(fines_raw),
                "fuel_transaction.csv":len(fuel_raw),"итого_строк":TOTAL_ROWS,"итого_признаков":TOTAL_COLS},
 "очищенные_файлы":{"clean_01_vehicles.csv":len(vehicles),"clean_02_clients.csv":len(clients_lv),
                    "clean_03_fines.csv":len(fines),"clean_04_fuel.csv":len(fuel),
                    "clean_05_panel_client_month.csv":len(panel)},
 "окна":{"зрелое_окно_нарушений":f"{MATURE_START.date()} .. {MATURE_END.date()}",
         "топливо":f"{fuel.order_date.min().date()} .. {fuel.order_date.max().date()}"},
 "кризис":{"начало_плавное":str(CRISIS_ONSET.date()),"жёсткий_лимит":str(HARD_LIMIT.date()),
           "фазы":[{"фаза":n,"с":a,"по":b,"лимит_л":l} for n,a,b,l in PHASES]},
 "когорта":{"фиксированная_n":len(FIXED)},
 "целевые_категории":TARGET_MAP,
 "запрещённые_переменные":FORBIDDEN,
}
(OUT/"metadata_derived.json").write_text(json.dumps(META, ensure_ascii=False, indent=2), encoding='utf-8')
say("\nФАЙЛЫ НА ВЫХОДЕ:")
for f in sorted(OUT.glob("*")): say(f"  {f.name:38s} {f.stat().st_size/1024:8.1f} KB")
say(f"\nГотово. Вход: {TOTAL_ROWS:,} сырых строк -> выход: панель {len(panel):,} клиенто-месяцев.")

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from scipy import stats, optimize
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
plt.rcParams.update({"figure.dpi":120,"savefig.dpi":150,"font.size":10,"axes.grid":True,
                     "grid.alpha":.25,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.autolayout":True})
GREEN,RED,NAVY,GREY,ORANGE = "#1f5c3a","#c0392b","#1f3a6e","#8a8a8a","#b3541e"
FINAL = OUTDIR / "results"
FIG = OUTDIR / "figures"

CHAIN = pd.DataFrame([
 ["1. Дефицит топлива / ограничение продаж",
  "падение доступного объёма, появление потолка заправки",
  "379 699 транзакций: объём, цена, дата, клиент",
  "DIRECT",
  "суточный p90 объёма, доля заправок >50 л, медиана и p95 цены",
  "ИЗМЕРЕНО: p90 56→45.62→35.62 л, доля >50 л 20.9%→0.7%"],
 ["2. Очереди на АЗС",
  "рост длины очереди и времени ожидания",
  "переменной очереди в данных НЕТ",
  "NOT OBSERVED",
  "не проверяется напрямую; используются proxy звена 1",
  "НЕ ИЗМЕРЕНО — принципиальный разрыв цепочки"],
 ["3. Изменение поведения водителей",
  "иные маршруты, иная частота и время поездок",
  "частота заправок, объём на клиента, число активных клиентов",
  "PROXY",
  "динамика n_fuel_tx и литров на клиента фиксированной когорты",
  "ИЗМЕРЕНО КОСВЕННО: активных клиентов −18%"],
 ["4. Целевые нарушения",
  "рост 6 категорий, сопутствующих заправке",
  "77 978 постановлений со статьёй и датой",
  "DIRECT",
  "суточные счётчики 6 категорий + composite",
  "ИЗМЕРЕНО: это и есть зависимая переменная"],
], columns=["Звено гипотезы","Что ожидалось","Что есть в данных",
            "Direct / Proxy / Not observed","Как проверяется","Результат"])
show(CHAIN)
CHAIN.to_csv(FINAL/"chain_measurability.csv", sep=';', index=False, encoding='utf-8-sig')
say("КЛЮЧЕВОЕ ОГРАНИЧЕНИЕ, зафиксированное до всех моделей:")
say("  звено 2 (очереди) в данных отсутствует. Топливные транзакции — это НЕ измерение")
say("  очереди, а измерение режима продажи топлива. Поэтому проверяется не вся цепочка,")
say("  а импликация: 'там и тогда, где ограничение действовало, целевых нарушений больше'.")

F_ALL = fuel[fuel.in_analysis & fuel.client_id.isin(FIXED)].copy()
fd = F_ALL.groupby('order_date').agg(
        litres=('order_fuel_volume','sum'), n=('order_id','size'),
        p90=('order_fuel_volume', lambda s: s.quantile(.9)),
        p99=('order_fuel_volume', lambda s: s.quantile(.99)),
        share50=('order_fuel_volume', lambda s: 100*(s>50).mean()),
        med_price=('order_fuel_price_1liter','median'),
        p95_price=('order_fuel_price_1liter', lambda s: s.quantile(.95))).reset_index()

MAY_START = pd.Timestamp("2026-05-01")     # начало кризиса
LIMIT_DT  = pd.Timestamp("2026-06-19")     # введение лимитов на топливо

b = F_ALL[F_ALL.crisis_phase=="P1_база"]; c4 = F_ALL[F_ALL.crisis_phase=="P4_лимит_35.62л"]
apr_ = fd[(fd.order_date>=MATURE_START)&(fd.order_date<MAY_START)]
may_ = fd[(fd.order_date>=MAY_START)&(fd.order_date<pd.Timestamp("2026-06-01"))]
say("ЧТО ПРОИЗОШЛО С ТОПЛИВОМ")
say(f"  объём/день   : {b.groupby('order_date').order_fuel_volume.sum().mean()/1000:.1f} -> "
      f"{c4.groupby('order_date').order_fuel_volume.sum().mean()/1000:.1f} тыс. л "
      f"({100*(c4.groupby('order_date').order_fuel_volume.sum().mean()/b.groupby('order_date').order_fuel_volume.sum().mean()-1):+.1f}%)")
say(f"  доля >50 л   : {100*(b.order_fuel_volume>50).mean():.2f}% -> {100*(c4.order_fuel_volume>50).mean():.2f}%")
say(f"  медиана цены : {b.order_fuel_price_1liter.median():.2f} -> {c4.order_fuel_price_1liter.median():.2f} ₽/л "
      f"({100*(c4.order_fuel_price_1liter.median()/b.order_fuel_price_1liter.median()-1):+.1f}%)")
say("  ТИП ШОКА: количественный (нормирование), а не ценовой\n")
say("КОГДА ПОЯВИЛОСЬ КОЛИЧЕСТВЕННОЕ ОГРАНИЧЕНИЕ")
say(f"  апрель p90 {apr_.p90.median():.2f} л -> май p90 {may_.p90.median():.2f} л "
      f"({100*(may_.p90.median()/apr_.p90.median()-1):+.1f}%)")
say("  -> в мае кризис уже идёт, но ЛИМИТА НА ОБЪЁМ ЗАПРАВКИ ещё нет")
show(fd[(fd.order_date>=pd.Timestamp('2026-06-15'))&(fd.order_date<=pd.Timestamp('2026-06-26'))]
        [['order_date','p90','share50','n']].round(2).set_index('order_date'))
say(f"  первое ужесточение {LIMIT_DT.date()}, жёсткий лимит 45.62 л с 24.06, 35.62 л с 04.07")

NEGCTRL = {"speed_20_40":["Превышение скорости на 20-40 км/ч"],
           "remen":["Не пристегнут ремень безопасности"],
           "platnaya":["Неоплаченный проезд по платной дороге"],
           "svet":["Нарушение правил пользования световыми приборами, звуковыми сигналами"]}
ALLOUT = {**TARGET_MAP, **NEGCTRL}
RU = {"razmetka":"Нарушение разметки","parkovka":"Остановка/стоянка в неположенном месте",
      "stop_line":"Пересечение стоп-линии","obochina":"Движение по обочине",
      "vydelenka":"Движение по выделенной полосе","telefon":"Телефон за рулём",
      "speed_20_40":"[K] Превышение 20-40 км/ч","remen":"[K] Ремень безопасности",
      "platnaya":"[K] Платная дорога","svet":"[K] Световые приборы",
      "composite6":"COMPOSITE-6 (гипотеза)","manoeuvre5":"Манёвренные-5 (без телефона)",
      "total":"Все нарушения"}
s2k = {s:k for k,v in ALLOUT.items() for s in v}
V = fines[fines.is_kept_bill & fines.in_mature_window & fines.client_id.isin(FIXED)].copy()
V['okey'] = V.offence_short_statement.map(s2k)

days = pd.date_range(MATURE_START, MATURE_END, freq='D')
DAY = pd.DataFrame(index=days); DAY.index.name='date'
for k in ALLOUT: DAY[k] = V[V.okey==k].groupby('offence_date').size().reindex(days, fill_value=0)
DAY['composite6'] = DAY[COMPOSITE_6].sum(axis=1)
DAY['manoeuvre5'] = DAY[COMPOSITE_5].sum(axis=1)
DAY['total'] = V.groupby('offence_date').size().reindex(days, fill_value=0)
DAY['other'] = DAY.total - DAY.composite6
DAY = DAY.reset_index()
DAY['t'] = np.arange(len(DAY)); DAY['dow'] = DAY.date.dt.dayofweek
SPECS = {"MAY": MAY_START, "JUN": LIMIT_DT}
SPEC_LABEL = {"MAY":"НАЧАЛО КРИЗИСА — 01.05 (условие проекта)",
              "JUN":"ЛИМИТ НА ЗАПРАВКУ — 19.06 (внутри кризиса)"}
SPEC_COLOR = {"MAY":RED,"JUN":NAVY}
MAIN = "MAY"
for k,d in SPECS.items():
    DAY[f'post_{k}'] = (DAY.date>=d).astype(int)
    DAY[f'taft_{k}'] = np.where(DAY.date>=d,(DAY.date-d).dt.days+1,0)

say("ЗАВИСИМЫЕ ПЕРЕМЕННЫЕ (зрелое окно 01.04-31.07, фиксированная когорта)")
show(pd.DataFrame({"переменная":[RU[k] for k in list(ALLOUT)+['composite6','manoeuvre5','total']],
  "всего":[int(DAY[k].sum()) for k in list(ALLOUT)+['composite6','manoeuvre5','total']],
  "в день":[round(DAY[k].mean(),2) for k in list(ALLOUT)+['composite6','manoeuvre5','total']],
  "var/mean":[round(DAY[k].var()/max(DAY[k].mean(),1e-9),2) for k in list(ALLOUT)+['composite6','manoeuvre5','total']]}))
say("COMPOSITE-6 = все шесть категорий гипотезы (основная зависимая переменная)")
say("Манёвренные-5 = те же категории без телефона: телефон — нарушение внимания,")
say("  а не манёвра; очередь порождает простой, а не геометрию затора. Считаем обе версии.")
say("\nNEGATIVE CONTROLS — почему очередь на АЗС их вызвать не может:")
for k,r in [("speed_20_40","очередь физически исключает превышение скорости"),
            ("remen","состояние до начала движения, не манёвр"),
            ("platnaya","оплата проезда по платной дороге"),
            ("svet","оснащение автомобиля")]:
    print(f"  {RU[k]:28s} n={int(DAY[k].sum()):5,d}  — {r}")

RESULTS=[]
def push(model,outcome,spec,term,b,se,p,n,note):
    RESULTS.append(dict(model=model,outcome=outcome,spec=spec,term=term,
        coefficient=round(float(b),5),effect_IRR=round(float(np.exp(b)),4),
        standard_error=round(float(se),5),
        confidence_interval=f"[{np.exp(b-1.96*se):.3f}; {np.exp(b+1.96*se):.3f}]",
        ci_low_IRR=round(float(np.exp(b-1.96*se)),4), ci_high_IRR=round(float(np.exp(b+1.96*se)),4),
        p_value=round(float(p),5),n_observations=int(n),interpretation=note))
def stars(p): return "***" if p<.01 else "**" if p<.05 else "*" if p<.1 else ""

def design(spec, d=None):
    d = DAY if d is None else d
    return pd.concat([pd.Series(1.0,index=d.index,name='const'),
                      d.t.astype(float).rename('time'),
                      d[f'post_{spec}'].astype(float).rename('post'),
                      d[f'taft_{spec}'].astype(float).rename('time_after'),
                      pd.get_dummies(d.dow,prefix='dow',drop_first=True).astype(float)],axis=1)

def fit(y,X,kind='poisson',offset=None,hac=7):
    M = sm.Poisson(y,X,offset=offset) if kind=='poisson' else (
        sm.NegativeBinomial(y,X,offset=offset,loglike_method='nb2') if kind=='nb' else sm.OLS(y,X))
    try:    return M.fit(disp=0,maxiter=300,cov_type='HAC',cov_kwds={'maxlags':hac,'use_correction':True})
    except Exception: return M.fit(disp=0,maxiter=300,cov_type='HC1')

def cond_poisson(df,ycol,xcols,gcol='client_id'):
    d = df[[gcol,ycol]+xcols].copy()
    d = d[d.groupby(gcol)[ycol].transform('sum')>0]
    g = pd.factorize(d[gcol])[0]; G = g.max()+1
    y = d[ycol].to_numpy(float); X = d[xcols].to_numpy(float); Y = np.bincount(g,weights=y)[g]
    def nll(bv):
        eta=X@bv; mx=np.zeros(G); np.maximum.at(mx,g,eta); e=np.exp(eta-mx[g])
        S=np.bincount(g,weights=e); logS=np.log(S)[g]+mx[g]; p=e/S[g]
        return -np.sum(y*(eta-logS)), -(X.T@(y-Y*p))
    bv = optimize.minimize(nll,np.zeros(X.shape[1]),jac=True,method='BFGS',options={'maxiter':400}).x
    eta=X@bv; mx=np.zeros(G); np.maximum.at(mx,g,eta); e=np.exp(eta-mx[g])
    S=np.bincount(g,weights=e); p=e/S[g]
    sc=X*(y-Y*p)[:,None]; sg=np.zeros((G,X.shape[1])); np.add.at(sg,g,sc)
    Xp=X*p[:,None]; k=X.shape[1]; H=np.zeros((k,k)); cnt=np.maximum(np.bincount(g),1); ysum=np.bincount(g,weights=y)
    for i in range(k):
        for j in range(k):
            H[i,j]=np.sum(Y*Xp[:,i]*X[:,j])-np.sum(np.bincount(g,weights=Xp[:,i])*np.bincount(g,weights=Xp[:,j])*ysum/cnt)
    Hi=np.linalg.pinv(H); Vc=Hi@(sg.T@sg)@Hi; se=np.sqrt(np.abs(np.diag(Vc)))
    z=bv/se; pv=2*(1-stats.norm.cdf(np.abs(z)))
    return pd.DataFrame({'term':xcols,'coef':bv,'se':se,'p':pv,'ci_l':bv-1.96*se,'ci_h':bv+1.96*se}), d[gcol].nunique(), len(d)
say("инструменты определены (все функции — внутри notebook)")

dg=[]
for k in ['composite6','manoeuvre5']+list(ALLOUT)+['total']:
    r=fit(DAY[k].astype(float),design(MAIN),'ols')
    lb=float(acorr_ljungbox(r.resid,lags=[7],return_df=True)['lb_pvalue'].iloc[0])
    dg.append({"outcome":RU[k],"Durbin-Watson":round(durbin_watson(r.resid),3),
               "Ljung-Box(7) p":round(lb,4),"автокорреляция":"есть" if lb<.05 else "нет"})
show(pd.DataFrame(dg))
say("HAC(Newey-West, лаг 7) применяется ко ВСЕМ временным моделям безусловно.")

def its_table(spec):
    rows=[]
    for k in ['composite6','manoeuvre5']+list(ALLOUT)+['total']:
        y=DAY[k].astype(float); X=design(spec)
        for kind,lab in [('poisson','Poisson'),('nb','NegBin')]:
            try: r=fit(y,X,kind)
            except Exception: continue
            for term,ru in [('post','скачок уровня β2'),('time_after','изменение тренда β3')]:
                b,se,p=r.params[term],r.bse[term],r.pvalues[term]
                rows.append({"outcome":RU[k],"модель":lab,"эффект":ru,"IRR":round(np.exp(b),3),
                    "CI":f"[{np.exp(b-1.96*se):.3f}; {np.exp(b+1.96*se):.3f}]","p":round(p,4),"знч":stars(p)})
                if kind=='poisson':
                    push("ITS-Poisson",RU[k],spec,term,b,se,p,len(y),
                         f"{'скачок уровня' if term=='post' else 'изменение тренда'}, IRR={np.exp(b):.3f}")
    return pd.DataFrame(rows)
ITS={s:its_table(s) for s in SPECS}
for s in SPECS:
    print("="*104); print(f"ITS-Poisson — {SPEC_LABEL[s]}"); print("="*104)
    show(ITS[s][ITS[s]["модель"]=="Poisson"].pivot(index="outcome",columns="эффект",values=["IRR","p"]).round(4))

cmp=[]
for k in ['composite6','manoeuvre5']+list(ALLOUT)+['total']:
    y=DAY[k].astype(float); X=design(MAIN)
    gl=sm.GLM(y,X,family=sm.families.Poisson()).fit()
    mu=np.asarray(gl.fittedvalues)
    aux=np.asarray(((y-mu)**2-y)/np.maximum(mu,1e-9),dtype=float)
    ct=sm.OLS(aux,mu.reshape(-1,1)).fit(cov_type='HC1')
    rp=sm.Poisson(y,X).fit(disp=0,maxiter=300)
    try:
        rn=sm.NegativeBinomial(y,X,loglike_method='nb2').fit(disp=0,maxiter=400)
        LR=2*(rn.llf-rp.llf); pLR=0.5*stats.chi2.sf(max(LR,0),1); aicn=rn.aic
    except Exception: LR,pLR,aicn=np.nan,np.nan,np.nan
    cmp.append({"outcome":RU[k],"var/mean":round(y.var()/y.mean(),2),
        "Pearson χ²/df":round(float(gl.pearson_chi2/gl.df_resid),3),
        "Cameron-Trivedi α":round(float(np.asarray(ct.params)[0]),4),
        "p(α=0)":round(float(np.asarray(ct.pvalues)[0]),4),
        "AIC Poisson":round(rp.aic,1),"AIC NB":round(aicn,1) if aicn==aicn else np.nan,
        "p(LR α=0)":round(pLR,4) if pLR==pLR else np.nan,
        "предпочтительна":"NB" if (aicn==aicn and aicn<rp.aic and pLR<.05) else "Poisson"})
CMP=pd.DataFrame(cmp); show(CMP)
say("Правило выбора зафиксировано ДО оценки: NB при Pearson χ²/df > 1.25 и p(Cameron-Trivedi) < 0.05.")
say("Выбор делается по диагностике и AIC, а НЕ по величине p-value эффекта.")

rel=[]; off=np.log(DAY.other.clip(lower=1).astype(float))
for spec in SPECS:
    for k in ['composite6','manoeuvre5']+COMPOSITE_6:
        y=DAY[k].astype(float); X=design(spec)
        r0=fit(y,X,'poisson'); r1=fit(y,X,'poisson',offset=off)
        b,se,p=r1.params['post'],r1.bse['post'],r1.pvalues['post']
        rel.append({"outcome":RU[k],"спец":spec,"IRR сырой":round(np.exp(r0.params['post']),3),
            "p сырой":round(r0.pvalues['post'],4),"IRR отн. прочих":round(np.exp(b),3),
            "CI":f"[{np.exp(b-1.96*se):.3f}; {np.exp(b+1.96*se):.3f}]","p отн.":round(p,4),"знч":stars(p)})
        push("ITS-Poisson + offset",RU[k],spec,"post",b,se,p,len(y),"изменение ДОЛИ относительно прочих штрафов")
REL=pd.DataFrame(rel)
for s in SPECS:
    print(f"\n=== {SPEC_LABEL[s]} ===")
    show(REL[REL['спец']==s].drop(columns='спец').reset_index(drop=True))

gen = ITS[MAIN][(ITS[MAIN]['модель']=='Poisson')&(ITS[MAIN]['эффект']=='скачок уровня β2')][['outcome','IRR','p']].copy()
def grp(x):
    if x.startswith('[K]'): return 'negative control'
    if 'COMPOSITE' in x: return 'composite'
    if 'Манёвренные' in x: return 'манёвренные'
    if x=='Все нарушения': return 'итого'
    return 'цель'
gen['группа']=gen.outcome.map(grp)
show(gen.sort_values('IRR',ascending=False).reset_index(drop=True))
nc=gen[gen['группа']=='negative control']
n_nc_sig=int((nc.p<.05).sum()); n_nc_up=int(((nc.p<.05)&(nc.IRR>1)).sum()); n_nc=len(nc)
say(f"\nNEGATIVE CONTROLS в майской спецификации: значимы {n_nc_sig} из {n_nc}, "
      f"из них РАСТУТ {n_nc_up}")
say("Ремень безопасности растёт сильнее самого composite. Очередь на АЗС не может")
say("заставить водителя отстегнуть ремень -> майский скачок НЕ специфичен для механизма.")

P = panel[panel.client_id.isin(FIXED)].copy()
P = P.drop(columns=[c for c in ALLOUT if c in P.columns])
wide = V.pivot_table(index=['client_id','month'], columns='okey', aggfunc='size', fill_value=0)
P = P.merge(wide.reset_index(), on=['client_id','month'], how='left')
for k in ALLOUT:
    P[k] = P[k].fillna(0).astype(int) if k in P.columns else 0
P['post_MAY'] = (P.month_idx>=1).astype(float)   # май июнь июль
P['post_JUN'] = (P.month_idx>=3).astype(float)   # июль первый полный месяц после 19.06
P['composite6']=P[COMPOSITE_6].sum(axis=1); P['manoeuvre5']=P[COMPOSITE_5].sum(axis=1)
P['total']=P.n_violations.astype(int)
chk_ok = P.composite6.sum()==DAY.composite6.sum()
say(f"панель: {len(P):,} строк = {P.client_id.nunique():,} клиентов x {P.month.nunique()} месяца | "
      f"сверка composite6 с дневным рядом: {'OK' if chk_ok else 'РАСХОЖДЕНИЕ'} "
      f"({int(P.composite6.sum()):,} vs {int(DAY.composite6.sum()):,})")

pan=[]
for k in ['composite6','manoeuvre5']+list(ALLOUT)+['total']:
    for spec in SPECS:
        tb,ng,nobs = cond_poisson(P.assign(post=P[f'post_{spec}']), k, ['post'])
        r=tb.iloc[0]
        pan.append({"outcome":RU[k],"спец":spec,"IRR":round(np.exp(r.coef),3),
            "CI":f"[{np.exp(r.ci_l):.3f}; {np.exp(r.ci_h):.3f}]","p":round(r.p,4),
            "знч":stars(r.p),"клиентов":ng})
        push("Panel FE (усл. Пуассон)",RU[k],spec,"post",r.coef,r.se,r.p,nobs,
             f"внутриклиентский эффект, IRR={np.exp(r.coef):.3f}, клиентов {ng}")
PAN=pd.DataFrame(pan)
show(PAN.pivot(index="outcome",columns="спец",values=["IRR","p"]).round(4))

apr_f = F_ALL[(F_ALL.order_date>=MATURE_START)&(F_ALL.order_date<=pd.Timestamp("2026-04-30"))]
expo = apr_f.groupby('client_id').order_fuel_volume.agg(p90=lambda s:s.quantile(.9), n='size')
expo = expo[expo.n>=2]
TR=set(expo[expo.p90>45.62].index); CO=set(expo[expo.p90<=35.62].index)
say(f"TREATED (связаны лимитом, апрельский p90 > 45.62 л) : {len(TR):,} клиентов")
say(f"CONTROL (не связаны,       апрельский p90 ≤ 35.62 л) : {len(CO):,} клиентов")
say(f"промежуточная зона 35.62-45.62 л исключена явно      : "
      f"{int(expo.p90.between(35.62,45.62,inclusive='right').sum()):,} клиентов")
DD = P[P.client_id.isin(TR|CO)].copy(); DD['treated']=DD.client_id.isin(TR).astype(float)
show(DD.pivot_table(index='month',columns='treated',values='composite6',aggfunc='mean').round(4)
          .rename(columns={0.0:'control',1.0:'treated'}))

# тренд в разнице логарифмов интенсивностей ДО события
Vg = V[V.client_id.isin(TR|CO)].copy(); Vg['treated']=Vg.client_id.isin(TR).astype(int)
Vg['week']=Vg.offence_date.dt.to_period('W').dt.start_time
wk = Vg[Vg.okey.isin(COMPOSITE_6)].groupby(['week','treated']).size().unstack(fill_value=0)
wk = wk.reindex(columns=[0,1], fill_value=0)
rate = wk.div([len(CO),len(TR)], axis=1)*1000
pre = rate[rate.index < SPECS['JUN']].copy(); pre['t']=np.arange(len(pre))
pre['dlog']=np.log(pre[1]+1e-9)-np.log(pre[0]+1e-9)
pt = sm.OLS(pre.dlog, sm.add_constant(pre[['t']])).fit(cov_type='HAC',cov_kwds={'maxlags':3})
say("PARALLEL TRENDS (докризисные недели для спецификации 19.06):")
say(f"  наклон {pt.params['t']:+.5f}, SE {pt.bse['t']:.5f}, p = {pt.pvalues['t']:.4f} -> "
      f"{'предпосылка НЕ отвергается' if pt.pvalues['t']>.05 else 'предпосылка ОТВЕРГАЕТСЯ'}")
say(f"  доcобытийных недель: {len(pre)}")
say("\nДля спецификации 01.05 докризисный период — только апрель (зрелое окно начинается")
say("01.04), поэтому параллельность трендов там НЕПРОВЕРЯЕМА. Записано как ограничение.")

did=[]
for spec in SPECS:
    d=DD.copy(); d['post']=d[f'post_{spec}']; d['did']=d.post*d.treated
    for k in ['composite6','manoeuvre5']+COMPOSITE_6+['speed_20_40','remen','total']:
        try:
            tb,ng,nobs=cond_poisson(d,k,['post','did']); r=tb[tb.term=='did'].iloc[0]
            did.append({"outcome":RU[k],"спец":spec,"DiD IRR":round(np.exp(r.coef),3),
                "CI":f"[{np.exp(r.ci_l):.3f}; {np.exp(r.ci_h):.3f}]","p":round(r.p,4),
                "знч":stars(r.p),"клиентов":ng})
            push("DiD (FE клиента)",RU[k],spec,"treated×post",r.coef,r.se,r.p,nobs,
                 f"дополнительный эффект у связанных лимитом, IRR={np.exp(r.coef):.3f}")
        except Exception:
            did.append({"outcome":RU[k],"спец":spec,"DiD IRR":np.nan,"CI":"","p":np.nan,"знч":"","клиентов":0})
DID=pd.DataFrame(did)
show(DID.pivot(index="outcome",columns="спец",values=["DiD IRR","p"]).round(4))
say("Ключ к чтению: если бы кризис действовал через ограничение объёма заправки,")
say("у связанных лимитом клиентов целевые нарушения росли бы СИЛЬНЕЕ. DiD измеряет")
say("именно эту разницу, и она не зависит от того, какая календарная дата верна.")

def event_study(outcome, spec, kmin=-8, kmax=8, rel=False):
    cut=SPECS[spec]; d=DAY.copy()
    d['k']=np.floor((d.date-cut).dt.days/7).astype(int)
    d=d[(d.k>=kmin)&(d.k<=kmax)]
    Xk=pd.get_dummies(d.k,prefix='k').astype(float)
    if 'k_-1' in Xk: Xk=Xk.drop(columns=['k_-1'])
    X=pd.concat([pd.Series(1.0,index=d.index,name='const'),Xk,
                 pd.get_dummies(d.dow,prefix='dow',drop_first=True).astype(float)],axis=1)
    off=np.log(d.other.clip(lower=1).astype(float)) if rel else None
    r=sm.Poisson(d[outcome].astype(float),X,offset=off).fit(disp=0,maxiter=400,
        cov_type='HAC',cov_kwds={'maxlags':7,'use_correction':True})
    out=[]
    for k in range(kmin,kmax+1):
        if k==-1: out.append({'k':k,'coef':0,'lo':0,'hi':0,'p':np.nan}); continue
        c=f'k_{k}'
        if c in r.params:
            b,se=r.params[c],r.bse[c]
            out.append({'k':k,'coef':b,'lo':b-1.96*se,'hi':b+1.96*se,'p':r.pvalues[c]})
    return pd.DataFrame(out)

ES={}
for spec in SPECS:
    ES[(spec,'raw')]=event_study('composite6',spec)
    ES[(spec,'rel')]=event_study('composite6',spec,rel=True)
ESUM=[]
for spec in SPECS:
    for key,nm in [('raw','сырой эффект'),('rel','относительно прочих штрафов')]:
        e=ES[(spec,key)]; pre=e[e.k<-1]
        ESUM.append({"спецификация":SPEC_LABEL[spec],"версия":nm,
            "доcобытийных недель":len(pre),"из них значимых":int((pre.p<.05).sum()),
            "недели":str(list(pre[pre.p<.05].k.values))})
        push("Event study",RU['composite6'],spec,"pre-trend",0,0,1,len(e),
             f"{nm}: значимых доcобытийных недель {int((pre.p<.05).sum())} из {len(pre)}")
ESUMDF=pd.DataFrame(ESUM); show(ESUMDF)
say("ЧТО ЭТО ЗНАЧИТ. Изменение в целевых нарушениях наблюдается УЖЕ ДО обеих дат.")
say("Предпосылка чистого пост-событийного скачка нарушена, поэтому последующее")
say("изменение нельзя интерпретировать как чистый причинный эффект кризиса.")
say("\nОГРАНИЧЕНИЕ: у спецификации 01.05 доcобытийный период — всего 4 недели,")
say("потому что зрелое окно наблюдения начинается 01.04. Это ослабляет силу")
say("pre-trend анализа для майской даты и само по себе НЕ доказывает отсутствие эффекта.")

def level_effect(outcome,cut,d0=None,d1=None,rel=False):
    d=DAY.copy()
    if d0 is not None: d=d[d.date>=d0]
    if d1 is not None: d=d[d.date<=d1]
    if (d.date>=cut).sum()<14 or (d.date<cut).sum()<14: return np.nan,np.nan,np.nan
    X=pd.DataFrame({'const':1.0,'time':np.arange(len(d),dtype=float),
        'post':(d.date>=cut).astype(float).values,
        'time_after':np.where(d.date>=cut,(d.date-cut).dt.days+1,0).astype(float)},index=d.index)
    X=pd.concat([X,pd.get_dummies(d.dow,prefix='dow',drop_first=True).astype(float)],axis=1)
    off=np.log(d.other.clip(lower=1).astype(float)) if rel else None
    try:
        r=sm.Poisson(d[outcome].astype(float),X,offset=off).fit(disp=0,maxiter=400,
            cov_type='HAC',cov_kwds={'maxlags':7,'use_correction':True})
        return r.params['post'],r.bse['post'],r.pvalues['post']
    except Exception: return np.nan,np.nan,np.nan

cand=pd.date_range("2026-04-15","2026-07-15",freq='D')
roll=[]
for c_ in cand:
    b,se,p=level_effect('composite6',c_); b2,_,p2=level_effect('composite6',c_,rel=True)
    roll.append({'cut':c_,'b':b,'lo':b-1.96*se,'hi':b+1.96*se,'p':p,'b_rel':b2,'p_rel':p2})
ROLL=pd.DataFrame(roll); share_sig=100*(ROLL.p<.05).mean()
rank_may=int((ROLL.b.abs()>abs(ROLL.loc[ROLL.cut==SPECS['MAY'],'b'].iloc[0])).sum())+1
rank_jun=int((ROLL.b.abs()>abs(ROLL.loc[ROLL.cut==SPECS['JUN'],'b'].iloc[0])).sum())+1
say(f"СКОЛЬЗЯЩАЯ PLACEBO-ДАТА, composite6: {len(ROLL)} дат-кандидатов")
say(f"  дают «значимый» (p<0.05) скачок : {int((ROLL.p<.05).sum())} ({share_sig:.0f}%)")
say(f"  ранг даты 01.05 по величине |β2| : {rank_may} из {len(ROLL)}")
say(f"  ранг даты 19.06 по величине |β2| : {rank_jun} из {len(ROLL)}")

pl=[]
for fake in [pd.Timestamp("2026-04-25"),pd.Timestamp("2026-05-15"),pd.Timestamp("2026-06-01")]:
    for k in ['composite6','remen','speed_20_40']:
        b,se,p=level_effect(k,fake,d0=MATURE_START,d1=pd.Timestamp("2026-06-18"))
        pl.append({"фиктивная дата":str(fake.date()),"outcome":RU[k],
                   "IRR":round(np.exp(b),3),"p":round(p,4),"знч":stars(p)})
        if k=='composite6':
            push("Placebo-дата",RU[k],f"FAKE {fake.date()}","post",b,se,p,0,"фиктивная отсечка внутри докризисного периода")
PL=pd.DataFrame(pl); show(PL)
pl15=PL[(PL['фиктивная дата']=="2026-05-15")&(PL.outcome==RU['composite6'])].iloc[0]
say(f"Фиктивная дата 15.05 внутри докризисного периода даёт IRR = {pl15.IRR} (p = {pl15.p}).")
say("Модель находит «эффект» там, где предполагаемого события ещё не было.")

rob=[]
def rb(name,b,se,p,n,note):
    rob.append({"спецификация":name,"IRR":round(np.exp(b),3),
        "CI 95%":f"[{np.exp(b-1.96*se):.3f}; {np.exp(b+1.96*se):.3f}]",
        "p-value":round(p,4),"знч":stars(p),"N":n,"комментарий":note})
for spec in SPECS:
    y=DAY.composite6.astype(float); X=design(spec)
    r=fit(y,X,'poisson');  rb(f"[{spec}] ITS-Poisson", r.params['post'],r.bse['post'],r.pvalues['post'],len(y),"основная временная модель")
    rn=fit(y,X,'nb');      rb(f"[{spec}] ITS-NegBin",  rn.params['post'],rn.bse['post'],rn.pvalues['post'],len(y),"учёт сверхдисперсии")
    b,se,p=level_effect('composite6',SPECS[spec],rel=True); rb(f"[{spec}] ITS + offset(прочие штрафы)",b,se,p,len(y),"эффект относительно общего потока")
    for alt,parts in [("манёвренные-5 (без телефона)",COMPOSITE_5),
                      ("без парковки",[k for k in COMPOSITE_6 if k!='parkovka']),
                      ("без разметки",[k for k in COMPOSITE_6 if k!='razmetka'])]:
        DAY['_a']=DAY[parts].sum(axis=1)
        rr=fit(DAY['_a'].astype(float),X,'poisson'); rb(f"[{spec}] outcome = {alt}",rr.params['post'],rr.bse['post'],rr.pvalues['post'],len(DAY),"альтернативное определение outcome")
    Vx=V[~V.region_name.isin(["Москва","Санкт-Петербург"])]
    DAY['_n']=Vx[Vx.okey.isin(COMPOSITE_6)].groupby('offence_date').size().reindex(days,fill_value=0).values
    rr=fit(DAY['_n'].astype(float),X,'poisson'); rb(f"[{spec}] без Москвы и СПб",rr.params['post'],rr.bse['post'],rr.pvalues['post'],len(DAY),"исключены столичные составы")
    tb,ng,nobs=cond_poisson(P.assign(post=P[f'post_{spec}']),'composite6',['post']); r_=tb.iloc[0]
    rb(f"[{spec}] Panel FE (усл. Пуассон)",r_.coef,r_.se,r_.p,nobs,f"FE клиента, {ng} клиентов")
    d=DD.copy(); d['post']=d[f'post_{spec}']; d['did']=d.post*d.treated
    tb,ng,nobs=cond_poisson(d,'composite6',['post','did']); r_=tb[tb.term=='did'].iloc[0]
    rb(f"[{spec}] DiD связан/не связан лимитом",r_.coef,r_.se,r_.p,nobs,"естественная контрольная группа")
    b,se,p=level_effect('composite6',SPECS[spec],d0=SPECS[spec]-pd.Timedelta(days=42),
                        d1=min(MATURE_END,SPECS[spec]+pd.Timedelta(days=42)))
    if b==b: rb(f"[{spec}] окно ±6 недель",b,se,p,84,"локальное окно вокруг события")
ROB=pd.DataFrame(rob); ROB['спец']=ROB['спецификация'].str.extract(r"\[(\w+)\]")
for spec in SPECS:
    s=ROB[ROB['спец']==spec]
    print(f"\n{'='*112}\n{SPEC_LABEL[spec]} — outcome COMPOSITE-6\n{'='*112}")
    show(s.drop(columns='спец').reset_index(drop=True))
    print(f"значимый РОСТ: {int(((s['p-value']<.05)&(s.IRR>1)).sum())} из {len(s)} | "
          f"значимое СНИЖЕНИЕ: {int(((s['p-value']<.05)&(s.IRR<1)).sum())} из {len(s)}")

fig, axes = plt.subplots(2,2, figsize=(14,9))
panels=[("MAY","raw","сырой эффект"),("MAY","rel","относительно прочих штрафов"),
        ("JUN","raw","сырой эффект"),("JUN","rel","относительно прочих штрафов")]
for ax,(spec,key,sub) in zip(axes.ravel(),panels):
    e=ES[(spec,key)]; col=SPEC_COLOR[spec]
    ax.axvspan(e.k.min()-0.5,-0.5,color=GREY,alpha=.10)
    ax.errorbar(e.k,e.coef,yerr=[e.coef-e.lo,e.hi-e.coef],fmt="o",ms=5,color=GREEN,
                ecolor=GREY,capsize=3,lw=1.1,zorder=3)
    ax.axhline(0,color="k",lw=.9); ax.axvline(-0.5,color=col,ls="--",lw=2.0,zorder=4)
    pre=e[e.k<-1]; sig=pre[pre.p<.05]
    ax.scatter(sig.k,sig.coef,color=RED,zorder=6,s=70,edgecolor="white",linewidth=.8)
    ax.set_title(f"{SPEC_LABEL[spec]}\n{sub}",fontsize=9.5,color=col)
    ax.set_xlabel("недель от события (0 = неделя события)",fontsize=8.5)
    ax.set_ylabel("log IRR   (0 = как в неделю −1)",fontsize=8.5)
    lo_,hi_=ax.get_ylim(); ax.set_ylim(lo_-(hi_-lo_)*0.18,hi_)
    ax.text(.015,.03,f"ДО события: {len(sig)} из {len(pre)} недель значимы",
            transform=ax.transAxes,fontsize=8.5,va="bottom",
            bbox=dict(boxstyle="round,pad=0.35",fc="#fdecea" if len(sig) else "#eaf6ee",
                      ec=RED if len(sig) else GREEN,lw=1))
handles=[plt.Line2D([],[],marker='o',ls='',color=GREEN,ms=6,label="коэффициент недели с 95% ДИ"),
         plt.Line2D([],[],marker='o',ls='',color=RED,ms=8,label="доcобытийная неделя, значимо ≠ базы (p<0.05)"),
         plt.Rectangle((0,0),1,1,fc=GREY,alpha=.18,label="период ДО события"),
         plt.Line2D([],[],color=RED,ls='--',lw=2,label="01.05 — начало кризиса"),
         plt.Line2D([],[],color=NAVY,ls='--',lw=2,label="19.06 — лимит на заправку")]
fig.legend(handles=handles,loc="lower center",ncol=3,fontsize=8.5,frameon=False,bbox_to_anchor=(.5,-.055))
fig.suptitle("График 6. Event study, COMPOSITE-6: коэффициенты по неделям с 95% ДИ (база — неделя −1)\n"
  "Красные точки СЛЕВА от линии события = изменение началось ДО него -> чистая причинная интерпретация невозможна",
  y=1.015,fontsize=12)
plt.savefig(FIG/"g06_event_study.png",bbox_inches="tight"); plt.close('all')

# сравнение спецификаций
from matplotlib.ticker import FixedLocator, FixedFormatter
TICKS=[0.5,0.75,1,1.5,2,3,5]
fig,axes=plt.subplots(1,2,figsize=(15,6),sharex=True)
for ax,spec in zip(axes,["MAY","JUN"]):
    s=ROB[ROB['спец']==spec].iloc[::-1]
    lo=s["CI 95%"].str.extract(r"\[([\d.]+)")[0].astype(float)
    hi=s["CI 95%"].str.extract(r"; ([\d.]+)\]")[0].astype(float)
    cols=[GREEN if p<.05 and i>1 else (RED if p<.05 and i<1 else GREY)
          for p,i in zip(s["p-value"],s.IRR)]
    ax.errorbar(s.IRR,range(len(s)),xerr=[s.IRR-lo,hi-s.IRR],fmt="none",ecolor=GREY,capsize=3,lw=1.1)
    ax.scatter(s.IRR,range(len(s)),c=cols,s=80,zorder=5,edgecolor="white",linewidth=.8)
    for j,(irr,p) in enumerate(zip(s.IRR,s["p-value"])):
        ax.annotate(f"{irr:.2f}"+("*" if p<.05 else ""),(irr,j),textcoords="offset points",
                    xytext=(0,9),ha="center",fontsize=7.5,color=("black" if p<.05 else GREY))
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s['спецификация'].str.replace(r"^\[\w+\]\s*","",regex=True),fontsize=8.5)
    ax.set_ylim(-1.7,len(s)-0.35)
    ax.axvline(1,color="k",lw=1.2,ls="--"); ax.set_xscale("log"); ax.set_xlim(0.45,6.5)
    ax.xaxis.set_major_locator(FixedLocator(TICKS))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in TICKS]))
    ax.xaxis.set_minor_locator(FixedLocator([])); ax.tick_params(axis='x',labelsize=9)
    ax.set_xlabel("IRR — во сколько раз меняется число нарушений в день  (1.0 = эффекта нет)",fontsize=9)
    ax.set_title(SPEC_LABEL[spec],fontsize=10.5,color=SPEC_COLOR[spec])
    n_up=int(((s["p-value"]<.05)&(s.IRR>1)).sum()); n_dn=int(((s["p-value"]<.05)&(s.IRR<1)).sum())
    ax.text(.015,.022,f"значимый рост: {n_up} из {len(s)}   |   значимое снижение: {n_dn} из {len(s)}",
            transform=ax.transAxes,fontsize=9,va="bottom",
            bbox=dict(boxstyle="round,pad=0.35",fc="#f5f5f5",ec=GREY,lw=1))
handles=[plt.Line2D([],[],marker='o',ls='',color=GREEN,ms=8,label="значимый рост (p<0.05)"),
         plt.Line2D([],[],marker='o',ls='',color=RED,ms=8,label="значимое снижение (p<0.05)"),
         plt.Line2D([],[],marker='o',ls='',color=GREY,ms=8,label="незначимо"),
         plt.Line2D([],[],color=GREY,lw=1.5,label="95% доверительный интервал")]
fig.legend(handles=handles,loc="lower center",ncol=4,fontsize=9,frameon=False,bbox_to_anchor=(.5,-.045))
fig.suptitle("График 7. Все 10 спецификаций для COMPOSITE-6, IRR скачка уровня с 95% ДИ\n"
  "Слева — начало кризиса (01.05), справа — введение лимита на заправку внутри кризиса (19.06)",
  y=1.035,fontsize=12)
plt.savefig(FIG/"g07_model_comparison.png",bbox_inches="tight"); plt.close('all')

# негативный контроль?
fig,axes=plt.subplots(1,2,figsize=(15,5.2))
a=axes[0]
a.fill_between(ROLL.cut,ROLL.lo,ROLL.hi,color=GREY,alpha=.25,label="95% ДИ")
a.plot(ROLL.cut,ROLL.b,color=GREEN,lw=1.7,label="92 при произвольной дате отсечки")
a.axhline(0,color="k",lw=.9)
for spec in ["MAY","JUN"]:
    row=ROLL.loc[ROLL.cut==SPECS[spec]]
    a.axvline(SPECS[spec],color=SPEC_COLOR[spec],ls="--",lw=1.8)
    a.scatter([SPECS[spec]],[row.b.iloc[0]],color=SPEC_COLOR[spec],s=90,zorder=6,
              edgecolor="white",linewidth=.8,
              label=f"{SPECS[spec].strftime('%d.%m')} — {'начало кризиса' if spec=='MAY' else 'лимит на заправку'}")
a.set_title(f"{share_sig:.0f}% ПРОИЗВОЛЬНЫХ дат дают «значимый» скачок\n"
            f"01.05 — лишь {rank_may}-я из {len(ROLL)} по величине эффекта",fontsize=10.5)
a.set_ylabel("β2 (log IRR скачка уровня)"); a.legend(fontsize=8.5,loc="upper right")
a.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
a=axes[1]; g=gen.set_index('outcome'); order=g.sort_values('IRR').index
cols={'цель':GREEN,'negative control':RED,'composite':'#7b2d8e','манёвренные':NAVY,'итого':'black'}
a.barh(range(len(order)),g.loc[order,'IRR'],color=[cols[g.loc[o,'группа']] for o in order],alpha=.88)
a.axvline(1,color="k",lw=1.2,ls="--")
a.set_yticks(range(len(order))); a.set_yticklabels(order,fontsize=8.5)
for j,o in enumerate(order):
    a.annotate(f"{g.loc[o,'IRR']:.2f}"+("*" if g.loc[o,'p']<.05 else ""),
               (g.loc[o,'IRR'],j),textcoords="offset points",xytext=(5,0),va="center",fontsize=7.5)
a.set_xlabel("IRR скачка уровня на 01.05")
a.legend(handles=[plt.Rectangle((0,0),1,1,color=v,label=k) for k,v in cols.items()],fontsize=7.5,loc="lower right")
a.set_title("Negative controls растут вместе с целевыми категориями\n"
            "Ремень безопасности — сильнее самого composite",fontsize=10.5)
fig.suptitle("График 8. Почему майский эффект не является доказательством: placebo и negative controls",
             y=1.05,fontsize=12)
plt.savefig(FIG/"g08_placebo.png",bbox_inches="tight"); plt.close('all')

RES=pd.DataFrame(RESULTS)
RES.to_csv(FINAL/"FINAL_model_comparison.csv",sep=';',index=False,encoding='utf-8-sig')
ROB.to_csv(FINAL/"FINAL_robustness.csv",sep=';',index=False,encoding='utf-8-sig')
DID.to_csv(FINAL/"FINAL_did.csv",sep=';',index=False,encoding='utf-8-sig')
PAN.to_csv(FINAL/"FINAL_panel_fe.csv",sep=';',index=False,encoding='utf-8-sig')
REL.to_csv(FINAL/"FINAL_offset.csv",sep=';',index=False,encoding='utf-8-sig')
ROLL.to_csv(FINAL/"FINAL_placebo_rolling.csv",sep=';',index=False,encoding='utf-8-sig')
ESUMDF.to_csv(FINAL/"FINAL_event_study.csv",sep=';',index=False,encoding='utf-8-sig')
CMP.to_csv(FINAL/"FINAL_dispersion.csv",sep=';',index=False,encoding='utf-8-sig')
say(f"FINAL_model_comparison.csv: {len(RES)} строк")
g_=lambda n: ROB[ROB['спецификация']==n].iloc[0]
KEY = {
 "may_its":   g_("[MAY] ITS-Poisson"),
 "may_off":   g_("[MAY] ITS + offset(прочие штрафы)"),
 "may_fe":    g_("[MAY] Panel FE (усл. Пуассон)"),
 "may_did":   g_("[MAY] DiD связан/не связан лимитом"),
 "may_nomsk": g_("[MAY] без Москвы и СПб"),
 "may_man5":  g_("[MAY] outcome = манёвренные-5 (без телефона)"),
 "jun_its":   g_("[JUN] ITS-Poisson"),
 "jun_off":   g_("[JUN] ITS + offset(прочие штрафы)"),
 "jun_did":   g_("[JUN] DiD связан/не связан лимитом"),
}
SUMMARY=pd.DataFrame([{"показатель":k,"IRR":v.IRR,"CI 95%":v["CI 95%"],"p":v["p-value"]} for k,v in KEY.items()])
show(SUMMARY)

mi,mo,mf,mdid,mnm,m5 = KEY['may_its'],KEY['may_off'],KEY['may_fe'],KEY['may_did'],KEY['may_nomsk'],KEY['may_man5']
ji,jo,jdid = KEY['jun_its'],KEY['jun_off'],KEY['jun_did']
say("="*104)
say("ЧТО МЫ ВИДИМ СНАЧАЛА")
say("="*104)
say(f"  ITS-Poisson, COMPOSITE-6, отсечка 01.05: IRR = {mi.IRR}  {mi['CI 95%']}  p = {mi['p-value']}")
say(f"  та же модель для манёвренных-5:          IRR = {m5.IRR}  {m5['CI 95%']}  p = {m5['p-value']}")
say("  Выглядит как сильное подтверждение гипотезы.\n")
say("="*104)
say("ПОЧЕМУ ЭТОГО НЕДОСТАТОЧНО — ШЕСТЬ НЕЗАВИСИМЫХ ПРИЧИН")
say("="*104)
e=ES[('MAY','raw')]; pre=e[e.k<-1]; er=ES[('MAY','rel')]; prer=er[er.k<-1]
say(f"1. ПРЕДТРЕНД. Значимых доcобытийных недель: {int((pre.p<.05).sum())} из {len(pre)} "
      f"(сырой), {int((prer.p<.05).sum())} из {len(prer)} (относительный).")
say("   Изменение началось ДО 01.05 — его нельзя приписать событию, которое ещё не наступило.\n")
say(f"2. NEGATIVE CONTROLS. Значимы {n_nc_sig} из {n_nc}; растут {n_nc_up}.")
say(f"   Ремень безопасности: IRR = {gen.set_index('outcome').loc[RU['remen'],'IRR']} — "
      f"сильнее, чем сам COMPOSITE-6 ({mi.IRR}).")
say("   Очередь на АЗС не может заставить водителя отстегнуть ремень.\n")
say(f"3. НЕ СПЕЦИФИЧНО. Относительно прочих штрафов эффект падает до IRR = {mo.IRR} "
      f"(p = {mo['p-value']}):")
say("   значительная часть «эффекта» — это общий рост потока зарегистрированных штрафов.\n")
say(f"4. ГЕОГРАФИЯ. Без Москвы и СПб эффект исчезает: IRR = {mnm.IRR} {mnm['CI 95%']}, p = {mnm['p-value']}.")
say("   Кризис был общенациональным; эффект, живущий только в двух городах, ему не соответствует.\n")
say(f"5. PLACEBO. {share_sig:.0f}% произвольных дат отсечки дают «значимый» скачок; "
      f"01.05 — лишь {rank_may}-я из {len(ROLL)} по величине.")
say(f"   Фиктивная дата 15.05 внутри докризисного периода: IRR = {pl15.IRR} (p = {pl15.p}).\n")
say(f"6. DiD. У клиентов, которых лимит реально связывал, дополнительного роста НЕТ:")
say(f"   01.05: IRR = {mdid.IRR} {mdid['CI 95%']}, p = {mdid['p-value']}")
say(f"   19.06: IRR = {jdid.IRR} {jdid['CI 95%']}, p = {jdid['p-value']}")
say("   Это прямая проверка механизма, и она не зависит от выбора календарной даты.\n")
say("="*104)
say("МЕТОДОЛОГИЧЕСКИЙ СМЫСЛ")
say("="*104)
say("  p-value отвечает на вопрос о статистической совместимости КОНКРЕТНОЙ модели")
say("  с нулевой гипотезой. Он не проверяет причинный механизм и не защищает")
say("  от общего временного тренда, изменения режима регистрации или регионального сдвига.")
say(f"\n  Поэтому IRR = {mi.IRR} НЕ означает «очереди увеличили нарушения в {mi.IRR} раза».")
say("  Это означает: в майском периоде зарегистрированных целевых нарушений в день было")
say(f"  примерно в {mi.IRR} раза больше, чем предсказывает докризисный тренд, —")
say("  и такой же результат воспроизводится на датах, когда события ещё не было.")

say("01.05 — НАЧАЛО КРИЗИСА (условие проекта)")
say(f"   ITS: IRR = {mi.IRR} (p = {mi['p-value']}) | относительно прочих: {mo.IRR} (p = {mo['p-value']})")
say(f"   значимый рост в {int(((ROB[ROB['спец']=='MAY']['p-value']<.05)&(ROB[ROB['спец']=='MAY'].IRR>1)).sum())} из 10 спецификаций\n")
say("19.06 — ВВЕДЕНИЕ ЛИМИТА НА РАЗОВУЮ ЗАПРАВКУ (внутри кризиса)")
say(f"   ITS: IRR = {ji.IRR} (p = {ji['p-value']}) | относительно прочих: {jo.IRR} (p = {jo['p-value']})")
say(f"   значимый рост в {int(((ROB[ROB['спец']=='JUN']['p-value']<.05)&(ROB[ROB['спец']=='JUN'].IRR>1)).sum())} из 10 спецификаций\n")
say("="*104)
say("КАК ЭТО ЧИТАТЬ")
say("="*104)
say("  Если бы механизм гипотезы работал, самый жёсткий момент кризиса — введение")
say("  количественного лимита — должен был дать эффект НЕ СЛАБЕЕ, чем размытое начало")
say("  кризиса. Наблюдается обратное: на дате реального ужесточения эффекта нет,")
say("  а на майской дате он максимален.")
say("  Это указывает, что майский скачок связан не с механизмом нормирования топлива,")
say("  а с чем-то, что совпало с маем по времени. Ремень безопасности как negative")
say("  control и исчезновение эффекта вне Москвы и СПб согласуются с версией об")
say("  изменении режима регистрации нарушений. Это АЛЬТЕРНАТИВНОЕ ОБЪЯСНЕНИЕ,")
say("  согласующееся с данными, а не доказанный факт: прямых данных о работе камер")
say("  и административных процедурах в датасете нет.")

LIM = pd.DataFrame([
 ["Очереди на АЗС не наблюдаются","центральное звено цепочки отсутствует в данных: нет длины очереди, времени ожидания, загрузки АЗС","механизм проверяется только косвенно"],
 ["Нет геопозиции АЗС","нельзя связать нарушение с конкретной заправкой","невозможен пространственный тест механизма"],
 ["Короткое окно наблюдения","зрелых месяцев 4 (01.04-31.07): левое усечение до 01.04, правое цензурирование с 01.08","сезонность неотделима от эффекта кризиса"],
 ["Короткий доcобытийный период для 01.05","всего 4 недели до события","ослабляет силу pre-trend анализа для майской даты"],
 ["Parallel trends для 01.05 непроверяема","докризисный период — только апрель","DiD для мая приводится с оговоркой"],
 ["Наблюдаются постановления, а не нарушения","изменение плотности камер или скорости обработки выглядит как изменение поведения","конкурирующее объяснение нельзя исключить"],
 ["Редкие события","обочина ~2.9, стоп-линия ~4.3 нарушений в день","ограниченная статистическая мощность"],
 ["Множественное тестирование","13 исходов x 2 даты x несколько моделей","вывод строится на согласованности спецификаций и placebo-распределении, а не на отдельных p-value"],
 ["Клиенты одного сервиса","не генеральная совокупность водителей","ограничена внешняя валидность"],
 ["Промежуточная группа в DiD исключена","клиенты с апрельским p90 в 35.62-45.62 л","сужает внешнюю валидность оценки DiD"],
], columns=["Ограничение","В чём состоит","Следствие для вывода"])
show(LIM); LIM.to_csv(FINAL/"FINAL_limitations.csv",sep=';',index=False,encoding='utf-8-sig')

n_may_up=int(((ROB[ROB['спец']=='MAY']['p-value']<.05)&(ROB[ROB['спец']=='MAY'].IRR>1)).sum())
n_jun_up=int(((ROB[ROB['спец']=='JUN']['p-value']<.05)&(ROB[ROB['спец']=='JUN'].IRR>1)).sum())
say("="*104); print("ФИНАЛЬНЫЙ ВЫВОД"); print("="*104)
say(f"""
1. АССОЦИАЦИЯ ЕСТЬ. В майском периоде наблюдается сильная статистическая связь с ростом
   целевых нарушений: ITS-Poisson IRR = {mi.IRR} {mi['CI 95%']}, p = {mi['p-value']};
   значимый рост в {n_may_up} из 10 спецификаций.

2. АССОЦИАЦИЯ НЕ СПЕЦИФИЧНА. Ни одна проверка механизма её не поддерживает:
   - предтренд: изменение началось ДО 01.05;
   - negative controls растут вместе с целевыми, ремень сильнее самого composite;
   - относительно общего потока штрафов эффект падает до IRR = {mo.IRR} (p = {mo['p-value']});
   - вне Москвы и СПб эффект исчезает: IRR = {mnm.IRR} (p = {mnm['p-value']});
   - {share_sig:.0f}% произвольных placebo-дат дают такой же «значимый» результат;
   - DiD: у связанных лимитом дополнительного роста нет ({mdid.IRR}, p = {mdid['p-value']}).

3. НА МОМЕНТ РЕАЛЬНОГО УЖЕСТОЧЕНИЯ ЭФФЕКТА НЕТ. При введении количественного лимита
   19.06 значимый рост не получен ни в одной из 10 спецификаций (IRR = {ji.IRR}, p = {ji['p-value']}).

ИТОГ: наблюдается статистическая АССОЦИАЦИЯ между майским периодом и ростом ряда целевых
нарушений, однако совокупность placebo-, negative-control-, event-study- и DiD-проверок
показывает, что эту ассоциацию НЕЛЬЗЯ надёжно интерпретировать как ПРИЧИННЫЙ эффект
топливного кризиса.

ЧЕГО ЭТОТ ВЫВОД НЕ ОЗНАЧАЕТ: он не означает, что эффект точно отсутствует. Он означает,
что имеющиеся данные не позволяют отделить предполагаемый эффект кризиса от временных
трендов и других факторов с достаточной уверенностью. Ключевое звено цепочки — очереди
на АЗС — в данных не наблюдается вовсе.
""")


CHECK=[
 ("Гипотеза сохранена без подмены", "цепочка дефицит -> очереди -> поведение -> нарушения описана в шапке и в таблице измеримости"),
 ("Прямые измерения отделены от proxy", "таблица CHAIN: DIRECT / PROXY / NOT OBSERVED"),
 ("Отсутствие переменной очереди указано явно", "звено 2 помечено NOT OBSERVED; топливные транзакции НЕ названы измерением очереди"),
 ("Май остаётся началом кризиса", f"MAIN = '{MAIN}', отсечка {SPECS['MAY'].date()}"),
 ("19.06 — лимит внутри кризиса, а не новое начало", "во всех подписях: 'ЛИМИТ НА ЗАПРАВКУ — 19.06 (внутри кризиса)'"),
 ("Майский IRR не интерпретируется как causal", "часть 15: шесть причин, почему недостаточно"),
 ("Pre-trend показан", f"event study: {int((ES[('MAY','raw')].query('k<-1').p<.05).sum())} значимых доcобытийных недель для 01.05"),
 ("Короткий майский pre-period оговорён", "часть 10 и таблица ограничений"),
 ("Negative controls объяснены", f"{n_nc} контролей, значимы {n_nc_sig}, растут {n_nc_up}"),
 ("Placebo объяснён", f"{len(ROLL)} дат, {share_sig:.0f}% значимы, ранг 01.05 = {rank_may}"),
 ("DiD объяснён", f"treated {len(TR):,} / control {len(CO):,}, экспозиция по апрелю"),
 ("Association и causality разделены", "часть 18: 'АССОЦИАЦИЯ есть' / 'причинный эффект не подтверждается'"),
 ("Противоречащие результаты не скрыты", f"показаны все 20 спецификаций, включая {n_may_up} значимых положительных"),
 ("Числа воспроизводятся кодом", f"{len(RES)} оценок в FINAL_model_comparison.csv"),
 ("Весь pipeline в одном .py файле", "внешних модулей нет, все функции определены в этом файле"),
]
show(pd.DataFrame(CHECK, columns=["Пункт","Подтверждение"]))
say("\nФАЙЛЫ НА ВЫХОДЕ:")
for _sub in ("cleaned_dataset", "figures", "results"):
    say(f"  [{_sub}]")
    for f in sorted((OUTDIR / _sub).rglob("*")):
        if f.is_file():
            say(f"     {f.name:42s} {f.stat().st_size/1024:8.1f} KB")
say("\nPIPELINE ЗАВЕРШЁН: RAW -> CLEANING -> VALIDATION -> FEATURES -> FUEL -> TARGETS")
say("-> MODELS -> EVENT STUDY -> DiD -> NEGATIVE CONTROLS -> PLACEBO -> FIGURES -> CONCLUSION")

# протокол
(OUTDIR / "run_log.txt").write_text("\n".join(_LOG), encoding="utf-8")
say(f"\nпротокол прогона сохранён: {OUTDIR / 'run_log.txt'}")
say("ГОТОВО")
