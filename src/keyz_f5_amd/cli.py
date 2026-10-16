"""Explicit command-line tools. No microphone, web server or background startup."""
import argparse
import json
import sys
import uuid


def main():
    parser = argparse.ArgumentParser(description='Experimental F5 DirectML runtime; RX 6900 XT verified only.')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('adapters')
    prepare = commands.add_parser('prepare13')
    prepare.add_argument('--source', required=True); prepare.add_argument('--destination', required=True)
    verify = commands.add_parser('verify-assets'); verify.add_argument('--models', required=True)
    bundle = commands.add_parser('import-assets')
    for name in ('preprocess','transformer','decode','vocab','destination'): bundle.add_argument('--'+name,required=True)
    speak = commands.add_parser('speak')
    serve = commands.add_parser('worker', help='Persistent bounded JSON-lines stdin/stdout; not a network service')
    for command in (speak, serve):
        for name in ('models','reference','transcript','output-dir'): command.add_argument('--'+name,required=True)
        command.add_argument('--device', type=int, help='Explicit AMD DXGI adapter index; other GPUs unverified')
    speak.add_argument('--text', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'prepare13':
            from .prepare13 import prepare
            print(json.dumps(prepare(args.source,args.destination))); return
        if args.command == 'adapters':
            from .adapters import adapters
            print(json.dumps(adapters())); return
        if args.command in ('verify-assets','import-assets'):
            from .assets import verify_assets, import_assets
            result = verify_assets(args.models) if args.command == 'verify-assets' else import_assets(args.preprocess,args.transformer,args.decode,args.vocab,args.destination)
            print(json.dumps(result)); return
        from .engine import Engine
        engine = Engine('keyz', args.models,args.reference,args.transcript,args.output_dir,device_id=args.device)
        try:
            if args.command == 'speak':
                print(json.dumps(engine.synthesize({'reply_id':uuid.uuid4().hex,'text':args.text}))); return
            print(json.dumps({'ready':True,'steps':13,'adapter':engine.target}),flush=True)
            for _ in range(256):
                line = sys.stdin.readline(8193)
                if not line or len(line) > 8192: break
                try:
                    result = engine.synthesize(json.loads(line))
                    print(json.dumps(result),flush=True)
                except (ValueError,RuntimeError) as error:
                    print(json.dumps({'error':str(error)[:240]}),flush=True)
        finally: engine.close()
    except (OSError,ValueError,RuntimeError) as error:
        parser.exit(1, str(error)+'\n')


if __name__ == '__main__': main()
