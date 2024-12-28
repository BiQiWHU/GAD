@REM python main.py --root D:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains DEEPDR FGADR IDRID MESSIDOR RLDR --target-domains APTOS --output results/DG_swin/APTOS --backbone swint
@REM python main.py --root D:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS FGADR IDRID MESSIDOR RLDR --target-domains DEEPDR --output results/DG_swin/DEEPDR --backbone swint
@REM python main.py --root D:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR IDRID MESSIDOR RLDR --target-domains FGADR --output results/DG_swin/FGADR --backbone swint
@REM python main.py --root D:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS  DEEPDR FGADR MESSIDOR RLDR --target-domains IDRID --output results/DG_swin/IDRID --backbone swint
@REM python main.py --root D:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID RLDR --target-domains MESSIDOR --output results/DG_swin/MESSIDOR --backbone swint
@REM python main.py --root D:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID MESSIDOR --target-domains RLDR --output results/DG_swin/RLDR --backbone swint



@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains DEEPDR FGADR IDRID MESSIDOR RLDR --target-domains APTOS --output results/DG_swin_2w/APTOS --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS FGADR IDRID MESSIDOR RLDR --target-domains DEEPDR --output results/DG_swin_2w/DEEPDR --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR IDRID MESSIDOR RLDR --target-domains FGADR --output results/DG_swin_2w/FGADR --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS  DEEPDR FGADR MESSIDOR RLDR --target-domains IDRID --output results/DG_swin_2w/IDRID --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID RLDR --target-domains MESSIDOR --output results/DG_swin_2w/MESSIDOR --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID MESSIDOR --target-domains RLDR --output results/DG_swin_2w/RLDR --backbone swint_iw



@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains DEEPDR FGADR IDRID MESSIDOR RLDR --target-domains APTOS --output D:/Med/DGDR/results/DG_swin_trip_p_/APTOS --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS FGADR IDRID MESSIDOR RLDR --target-domains DEEPDR --output D:/Med/DGDR/results/DG_swin_trip_p_/DEEPDR --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR IDRID MESSIDOR RLDR --target-domains FGADR --output D:/Med/DGDR/results/DG_swin_trip_p_/FGADR --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS  DEEPDR FGADR MESSIDOR RLDR --target-domains IDRID --output D:/Med/DGDR/results/DG_swin_trip_p_/IDRID --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID RLDR --target-domains MESSIDOR --output D:/Med/DGDR/results/DG_swin_trip_p_/MESSIDOR --backbone swint_iw
@REM python main.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID MESSIDOR --target-domains RLDR --output D:/Med/DGDR/results/DG_swin_trip_p_/RLDR --backbone swint_iw










@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains DEEPDR FGADR IDRID MESSIDOR RLDR --target-domains APTOS --output results/DG_swin_2w/APTOS_eval --backbone swint_iw --model_path results/DG_swin_2w/APTOS
@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS FGADR IDRID MESSIDOR RLDR --target-domains DEEPDR --output results/DG_swin_2w/DEEPDR_eval --backbone swint_iw --model_path results/DG_swin_2w/DEEPDR
@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR IDRID MESSIDOR RLDR --target-domains FGADR --output results/DG_swin_2w/FGADR_eval --backbone swint_iw --model_path results/DG_swin_2w/FGADR
@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS  DEEPDR FGADR MESSIDOR RLDR --target-domains IDRID --output results/DG_swin_2w/IDRID_eval --backbone swint_iw --model_path results/DG_swin_2w/IDRID
@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID RLDR --target-domains MESSIDOR --output results/DG_swin_2w/MESSIDOR_eval --backbone swint_iw --model_path results/DG_swin_2w/MESSIDOR
@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID MESSIDOR --target-domains RLDR --output results/DG_swin_2w/RLDR_eval --backbone swint_iw --model_path results/DG_swin_2w/RLDR

@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains DEEPDR FGADR IDRID MESSIDOR RLDR --target-domains APTOS --output results/DG_swin_trip_p/APTOS_eval --backbone swint_iw --model_path results/DG_swin_trip_p/APTOS
@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS FGADR IDRID MESSIDOR RLDR --target-domains DEEPDR --output results/DG_swin_trip_p/DEEPDR_eval --backbone swint_iw --model_path results/DG_swin_trip_p/DEEPDR
@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR IDRID MESSIDOR RLDR --target-domains FGADR --output results/DG_swin_trip_p/FGADR_eval --backbone swint_iw --model_path results/DG_swin_trip_p/FGADR
@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS  DEEPDR FGADR MESSIDOR RLDR --target-domains IDRID --output results/DG_swin_trip_p/IDRID_eval --backbone swint_iw --model_path results/DG_swin_trip_p/IDRID
@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID RLDR --target-domains MESSIDOR --output results/DG_swin_trip_p/MESSIDOR_eval --backbone swint_iw --model_path results/DG_swin_trip_p/MESSIDOR
@REM python eval.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID MESSIDOR --target-domains RLDR --output results/DG_swin_trip_p/RLDR_eval --backbone swint_iw --model_path results/DG_swin_trip_p/RLDR

@REM mu sigma vis
@REM python eval_mu.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains DEEPDR FGADR IDRID MESSIDOR RLDR --target-domains APTOS --output results/DG_swin_trip_p/APTOS_eval --backbone swint_iw --model_path results/DG_swin_trip_p/APTOS
@REM python eval_mu.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS FGADR IDRID MESSIDOR RLDR --target-domains DEEPDR --output results/DG_swin_trip_p/DEEPDR_eval --backbone swint_iw --model_path results/DG_swin_trip_p/DEEPDR
@REM python eval_mu.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR IDRID MESSIDOR RLDR --target-domains FGADR --output results/DG_swin_trip_p/FGADR_eval --backbone swint_iw --model_path results/DG_swin_trip_p/FGADR
@REM python eval_mu.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS  DEEPDR FGADR MESSIDOR RLDR --target-domains IDRID --output results/DG_swin_trip_p/IDRID_eval --backbone swint_iw --model_path results/DG_swin_trip_p/IDRID
@REM python eval_mu.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID RLDR --target-domains MESSIDOR --output results/DG_swin_trip_p/MESSIDOR_eval --backbone swint_iw --model_path results/DG_swin_trip_p/MESSIDOR
@REM python eval_mu.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID MESSIDOR --target-domains RLDR --output results/DG_swin_trip_p/RLDR_eval --backbone swint_iw --model_path results/DG_swin_trip_p/RLDR
python eval_mu.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains DEEPDR FGADR IDRID MESSIDOR RLDR --target-domains APTOS --output results/DG_swin_trip/APTOS_eval --backbone swint_iw --model_path results/DG_swin_trip/APTOS
python eval_mu.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS FGADR IDRID MESSIDOR RLDR --target-domains DEEPDR --output results/DG_swin_trip/DEEPDR_eval --backbone swint_iw --model_path results/DG_swin_trip/DEEPDR
python eval_mu.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR IDRID MESSIDOR RLDR --target-domains FGADR --output results/DG_swin_trip/FGADR_eval --backbone swint_iw --model_path results/DG_swin_trip/FGADR
python eval_mu.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS  DEEPDR FGADR MESSIDOR RLDR --target-domains IDRID --output results/DG_swin_trip/IDRID_eval --backbone swint_iw --model_path results/DG_swin_trip/IDRID
python eval_mu.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID RLDR --target-domains MESSIDOR --output results/DG_swin_trip/MESSIDOR_eval --backbone swint_iw --model_path results/DG_swin_trip/MESSIDOR
python eval_mu.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID MESSIDOR --target-domains RLDR --output results/DG_swin_trip/RLDR_eval --backbone swint_iw --model_path results/DG_swin_trip/RLDR


@REM tsne vis
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains DEEPDR FGADR IDRID MESSIDOR RLDR --target-domains APTOS --output results/DG_swin_trip_p/APTOS_eval --backbone swint_iw --model_path results/DG_swin_trip_p/APTOS
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS FGADR IDRID MESSIDOR RLDR --target-domains DEEPDR --output results/DG_swin_trip_p/DEEPDR_eval --backbone swint_iw --model_path results/DG_swin_trip_p/DEEPDR
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR IDRID MESSIDOR RLDR --target-domains FGADR --output results/DG_swin_trip_p/FGADR_eval --backbone swint_iw --model_path results/DG_swin_trip_p/FGADR
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS  DEEPDR FGADR MESSIDOR RLDR --target-domains IDRID --output results/DG_swin_trip_p/IDRID_eval --backbone swint_iw --model_path results/DG_swin_trip_p/IDRID
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID RLDR --target-domains MESSIDOR --output results/DG_swin_trip_p/MESSIDOR_eval --backbone swint_iw --model_path results/DG_swin_trip_p/MESSIDOR
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID MESSIDOR --target-domains RLDR --output results/DG_swin_trip_p/RLDR_eval --backbone swint_iw --model_path results/DG_swin_trip_p/RLDR

@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains DEEPDR FGADR IDRID MESSIDOR RLDR --target-domains APTOS --output results/DG_swin_2w/APTOS_eval --backbone swint_iw --model_path results/DG_swin_2w/APTOS
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS FGADR IDRID MESSIDOR RLDR --target-domains DEEPDR --output results/DG_swin_2w/DEEPDR_eval --backbone swint_iw --model_path results/DG_swin_2w/DEEPDR
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR IDRID MESSIDOR RLDR --target-domains FGADR --output results/DG_swin_2w/FGADR_eval --backbone swint_iw --model_path results/DG_swin_2w/FGADR
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS  DEEPDR FGADR MESSIDOR RLDR --target-domains IDRID --output results/DG_swin_2w/IDRID_eval --backbone swint_iw --model_path results/DG_swin_2w/IDRID
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID RLDR --target-domains MESSIDOR --output results/DG_swin_2w/MESSIDOR_eval --backbone swint_iw --model_path results/DG_swin_2w/MESSIDOR
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID MESSIDOR --target-domains RLDR --output results/DG_swin_2w/RLDR_eval --backbone swint_iw --model_path results/DG_swin_2w/RLDR

@REM heatmap vis
@REM python eval_heat.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains DEEPDR FGADR IDRID MESSIDOR RLDR --target-domains APTOS --output results/DG_swin_2w/APTOS_eval --backbone swint_iw --model_path results/DG_swin_2w/APTOS
@REM python eval_heat.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS FGADR IDRID MESSIDOR RLDR --target-domains DEEPDR --output results/DG_swin_2w/DEEPDR_eval --backbone swint_iw --model_path results/DG_swin_2w/DEEPDR
@REM python eval_heat.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR IDRID MESSIDOR RLDR --target-domains FGADR --output results/DG_swin_2w/FGADR_eval --backbone swint_iw --model_path results/DG_swin_2w/FGADR
@REM python eval_heat.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS  DEEPDR FGADR MESSIDOR RLDR --target-domains IDRID --output results/DG_swin_2w/IDRID_eval --backbone swint_iw --model_path results/DG_swin_2w/IDRID
@REM python eval_heat.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID RLDR --target-domains MESSIDOR --output results/DG_swin_2w/MESSIDOR_eval --backbone swint_iw --model_path results/DG_swin_2w/MESSIDOR
@REM python eval_heat.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID MESSIDOR --target-domains RLDR --output results/DG_swin_2w/RLDR_eval --backbone swint_iw --model_path results/DG_swin_2w/RLDR

@REM tsne vis2
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains DEEPDR FGADR IDRID MESSIDOR RLDR --target-domains APTOS --output results/DG_swin_trip_p/APTOS_eval --backbone swint_iw --model_path results/DG_swin_trip_p/APTOS
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS FGADR IDRID MESSIDOR RLDR --target-domains DEEPDR --output results/DG_swin_trip_p/DEEPDR_eval --backbone swint_iw --model_path results/DG_swin_trip_p/APTOS
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR IDRID MESSIDOR RLDR --target-domains FGADR --output results/DG_swin_trip_p/FGADR_eval --backbone swint_iw --model_path results/DG_swin_trip_p/APTOS
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS  DEEPDR FGADR MESSIDOR RLDR --target-domains IDRID --output results/DG_swin_trip_p/IDRID_eval --backbone swint_iw --model_path results/DG_swin_trip_p/APTOS
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID RLDR --target-domains MESSIDOR --output results/DG_swin_trip_p/MESSIDOR_eval --backbone swint_iw --model_path results/DG_swin_trip_p/APTOS
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID MESSIDOR --target-domains RLDR --output results/DG_swin_trip_p/RLDR_eval --backbone swint_iw --model_path results/DG_swin_trip_p/APTOS

@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains DEEPDR FGADR IDRID MESSIDOR RLDR --target-domains APTOS --output results/DG_swin_2w/APTOS_eval --backbone swint_iw --model_path results/DG_swin_2w/APTOS
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS FGADR IDRID MESSIDOR RLDR --target-domains DEEPDR --output results/DG_swin_2w/DEEPDR_eval --backbone swint_iw --model_path results/DG_swin_2w/APTOS
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR IDRID MESSIDOR RLDR --target-domains FGADR --output results/DG_swin_2w/FGADR_eval --backbone swint_iw --model_path results/DG_swin_2w/APTOS
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS  DEEPDR FGADR MESSIDOR RLDR --target-domains IDRID --output results/DG_swin_2w/IDRID_eval --backbone swint_iw --model_path results/DG_swin_2w/APTOS
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID RLDR --target-domains MESSIDOR --output results/DG_swin_2w/MESSIDOR_eval --backbone swint_iw --model_path results/DG_swin_2w/APTOS
@REM python eval_tsne.py --root E:/Med/dataset/DGDR --algorithm GDRNet --dg_mode DG --source-domains APTOS DEEPDR FGADR IDRID MESSIDOR --target-domains RLDR --output results/DG_swin_2w/RLDR_eval --backbone swint_iw --model_path results/DG_swin_2w/APTOS