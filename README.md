# transcriber

Real time transcription on Windows powered by Whisper of OpenAI.

```powershell
c:\...\transcriber>python main.py -h
usage: Transcriber [-h] --input {microphone,speaker} [--model {tiny,base,small,medium,large}]
                   [--language {en,zh,de,es,ru,ko,fr,ja,pt,tr,pl,ca,nl,ar,sv,it,id,hi,fi,vi,he,uk,el,ms,cs,ro,da,hu,ta,no,th,ur,hr,bg,lt,la,mi,ml,cy,sk,te,fa,lv,bn,sr,az,sl,kn,et,mk,br,eu,is,hy,ne,mn,bs,kk,sq,sw,gl,mr,pa,si,km,sn,yo,so,af,oc,ka,be,tg,sd,gu,am,yi,lo,uz,fo,ht,ps,tk,nn,mt,sa,lb,my,bo,tl,mg,as,tt,haw,ln,ha,ba,jw,su,}]
                   [--task {transcribe,translate}] [--save_audio SAVE_AUDIO]

optional arguments:
  -h, --help            show this help message and exit
  --input {microphone,speaker}
                        The input device to use
  --model {tiny,base,small,medium,large}
                        Whisper model type
  --language {en,zh,de,es,ru,ko,fr,ja,pt,tr,pl,ca,nl,ar,sv,it,id,hi,fi,vi,he,uk,el,ms,cs,ro,da,hu,ta,no,th,ur,hr,bg,lt,la,mi,ml,cy,sk,te,fa,lv,bn,sr,az,sl,kn,et,mk,br,eu,is,hy,ne,mn,bs,kk,sq,sw,gl,mr,pa,si,km,sn,yo,so,af,oc,ka,be,tg,sd,gu,am,yi,lo,uz,fo,ht,ps,tk,nn,mt,sa,lb,my,bo,tl,mg,as,tt,haw,ln,ha,ba,jw,su,}
                        Language to transcribe. Defaults to multilingual
  --task {transcribe,translate}
                        Task to perform, transcribe or translate.
  --save_audio SAVE_AUDIO
                        whether to save audio to file; True by default
```