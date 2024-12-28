@REM python main.py --root D:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains APTOS --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR DDR EYEPACS --output results/ESDG_swin/APTOS --backbone swint
@REM python main.py --root D:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains DEEPDR --target-domains APTOS FGADR IDRID MESSIDOR RLDR DDR EYEPACS --output results/ESDG_swin/DEEPDR --backbone swint
@REM python main.py --root D:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains FGADR --target-domains APTOS DEEPDR IDRID MESSIDOR RLDR DDR EYEPACS --output results/ESDG_swin/FGADR --backbone swint
@REM python main.py --root D:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains IDRID --target-domains APTOS  DEEPDR FGADR MESSIDOR RLDR DDR EYEPACS --output results/ESDG_swin/IDRID --backbone swint
@REM python main.py --root D:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains MESSIDOR --target-domains APTOS DEEPDR FGADR IDRID RLDR DDR EYEPACS --output results/ESDG_swin/MESSIDOR --backbone swint
@REM python main.py --root D:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains RLDR --target-domains APTOS DEEPDR FGADR IDRID MESSIDOR DDR EYEPACS --output results/ESDG_swin/RLDR --backbone swint



@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains APTOS --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR DDR EYEPACS --output results/ESDG_swin_2w/APTOS --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains DEEPDR --target-domains APTOS FGADR IDRID MESSIDOR RLDR DDR EYEPACS --output results/ESDG_swin_2w/DEEPDR --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains FGADR --target-domains APTOS DEEPDR IDRID MESSIDOR RLDR DDR EYEPACS --output results/ESDG_swin_2w/FGADR --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains IDRID --target-domains APTOS  DEEPDR FGADR MESSIDOR RLDR DDR EYEPACS --output results/ESDG_swin_2w/IDRID --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains MESSIDOR --target-domains APTOS DEEPDR FGADR IDRID RLDR DDR EYEPACS --output results/ESDG_swin_2w/MESSIDOR --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains RLDR --target-domains APTOS DEEPDR FGADR IDRID MESSIDOR DDR EYEPACS --output results/ESDG_swin_2w/RLDR --backbone swint_iw



@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains APTOS --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR DDR EYEPACS --output D:/Med/DGDR/results/ESDG_swin_trip_p_/APTOS --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains DEEPDR --target-domains APTOS FGADR IDRID MESSIDOR RLDR DDR EYEPACS --output D:/Med/DGDR/results/ESDG_swin_trip_p_/DEEPDR --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains FGADR --target-domains APTOS DEEPDR IDRID MESSIDOR RLDR DDR EYEPACS --output D:/Med/DGDR/results/ESDG_swin_trip_p_/FGADR --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains IDRID --target-domains APTOS  DEEPDR FGADR MESSIDOR RLDR DDR EYEPACS --output D:/Med/DGDR/results/ESDG_swin_trip_p_/IDRID --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains MESSIDOR --target-domains APTOS DEEPDR FGADR IDRID RLDR DDR EYEPACS --output D:/Med/DGDR/results/ESDG_swin_trip_p_/MESSIDOR --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains RLDR --target-domains APTOS DEEPDR FGADR IDRID MESSIDOR DDR EYEPACS --output D:/Med/DGDR/results/ESDG_swin_trip_p_/RLDR --backbone swint_iw












@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains APTOS --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR DDR EYEPACS --output results/ESDG_swin_2w/APTOS_eval --backbone swint_iw --model_path results/ESDG_swin_2w/APTOS
@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains DEEPDR --target-domains APTOS FGADR IDRID MESSIDOR RLDR DDR EYEPACS --output results/ESDG_swin_2w/DEEPDR_eval --backbone swint_iw --model_path results/ESDG_swin_2w/DEEPDR
@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains FGADR --target-domains APTOS DEEPDR IDRID MESSIDOR RLDR DDR EYEPACS --output results/ESDG_swin_2w/FGADR_eval --backbone swint_iw --model_path results/ESDG_swin_2w/FGADR
@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains IDRID --target-domains APTOS  DEEPDR FGADR MESSIDOR RLDR DDR EYEPACS --output results/ESDG_swin_2w/IDRID_eval --backbone swint_iw --model_path results/ESDG_swin_2w/IDRID
@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains MESSIDOR --target-domains APTOS DEEPDR FGADR IDRID RLDR DDR EYEPACS --output results/ESDG_swin_2w/MESSIDOR_eval --backbone swint_iw --model_path results/ESDG_swin_2w/MESSIDOR
@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains RLDR --target-domains APTOS DEEPDR FGADR IDRID MESSIDOR DDR EYEPACS --output results/ESDG_swin_2w/RLDR_eval --backbone swint_iw --model_path results/ESDG_swin_2w/RLDR

python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains APTOS --target-domains DEEPDR FGADR IDRID MESSIDOR RLDR DDR EYEPACS --output results/ESDG_swin_trip_p/APTOS_eval --backbone swint_iw --model_path results/ESDG_swin_trip_p/APTOS
python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains DEEPDR --target-domains APTOS FGADR IDRID MESSIDOR RLDR DDR EYEPACS --output results/ESDG_swin_trip_p/DEEPDR_eval --backbone swint_iw --model_path results/ESDG_swin_trip_p/DEEPDR
python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains FGADR --target-domains APTOS DEEPDR IDRID MESSIDOR RLDR DDR EYEPACS --output results/ESDG_swin_trip_p/FGADR_eval --backbone swint_iw --model_path results/ESDG_swin_trip_p/FGADR
python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains IDRID --target-domains APTOS  DEEPDR FGADR MESSIDOR RLDR DDR EYEPACS --output results/ESDG_swin_trip_p/IDRID_eval --backbone swint_iw --model_path results/ESDG_swin_trip_p/IDRID
python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains MESSIDOR --target-domains APTOS DEEPDR FGADR IDRID RLDR DDR EYEPACS --output results/ESDG_swin_trip_p/MESSIDOR_eval --backbone swint_iw --model_path results/ESDG_swin_trip_p/MESSIDOR
python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode ESDG --source-domains RLDR --target-domains APTOS DEEPDR FGADR IDRID MESSIDOR DDR EYEPACS --output results/ESDG_swin_trip_p/RLDR_eval --backbone swint_iw --model_path results/ESDG_swin_trip_p/RLDR