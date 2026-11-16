#!/usr/bin/env python3
"""Fetch one chunk of ESM-2 3B's pinned fp32 shards into /opt/model.

Builds of these images die on the machine that builds them in or after any step
that moves about a GiB or more: connections to huggingface.co that open and then
move no bytes, and steps that die silently minutes in. So the two shards
facebook/esm2_t36_3B_UR50D ships cross the builder in eighty-six short steps, one
128 MiB byte range of a pinned file per RUN instruction, each verified against
the manifest below before its layer is committed (the pattern protbert 1.0.3
proved on this builder):

- a step that needs nothing finishes in seconds, and one the network stalls
  fails within four minutes, having owed the builder only that much;
- every layer the steps commit is 128 MiB, not the 9.3 GB a shard would be, so
  no step's diff is large enough to be the thing the builder dies on;
- and because these RUN instructions never change between builds, whatever
  a failed build leaves in the builder's layer cache the next build reuses,
  so progress survives a build that dies -- which one long step never could.

The two sources are the Hub and its mirror; each request is a byte range, so
an attempt that already moved bytes before the connection stalled is resumed,
not restarted. The wrapper assembles each shard's parts at load time and
checks the whole file's sha256 before torch sees it, so both shards are
pinned twice: per chunk here, whole at run time.
"""
import hashlib
import os
import socket
import sys
import time
import urllib.error
import urllib.request

MODEL_DIR = os.environ.get("ESM2_3B_CHUNK_DIR", "/opt/model")  # the override is for testing this script
REVISION = "476b639933c8baad5ad09a60ac1a87f987b656fc"

# facebook/esm2_t36_3B_UR50D @ REVISION, the ESM-2 3B model: its two fp32
# shards, as released, in this order.
SHARDS = (
    ("pytorch_model-00001-of-00002.bin", 9976735419, "0f971f11c449d21422aa982b791619c10351972992c735f4c3cd43fe09790412"),
    ("pytorch_model-00002-of-00002.bin", 1390347055, "7560b46fc383c691fb74b915b7d4bcef40d3df181447f16ba4b298845e308d0c"),
)

SOURCES = [
    ("huggingface.co", "https://huggingface.co/facebook/esm2_t36_3B_UR50D/resolve/%s/%%s" % REVISION),
    ("hf-mirror.com", "https://hf-mirror.com/facebook/esm2_t36_3B_UR50D/resolve/%s/%%s" % REVISION),
]

# One row per RUN step: which shard (an index into SHARDS), the byte range of
# that shard the step delivers, and the sha256 it must show before its layer is
# committed. Chunks 0-74 are shard 0, 75-85 shard 1.
CHUNKS = [
    (0, 0, 134217728, "5e25fc8d1b5a0f220357b74d98ba8086731cff1c8857a849e055957ba3a01802"),
    (0, 134217728, 134217728, "5738c6ba780c1d40b55c69f8a7dc4d40e6e032cf815fded26ff12c3475a7ade2"),
    (0, 268435456, 134217728, "2971e820f36b8cad81b428d77884d385fb56425ff1143e6905c33c0a59022771"),
    (0, 402653184, 134217728, "c4ee46b753ed15f4ec9d3e18bb5f91aed9a3586cd32fe3570a6a231d35bcfb18"),
    (0, 536870912, 134217728, "bae21f2b54302cada008d45e7664c91d7b51879ee73383448978fe0350f9b794"),
    (0, 671088640, 134217728, "cd48df48353c2318cbcdfa57fc5601878d8e3192ad1ede7ab7d7cda40baadb2e"),
    (0, 805306368, 134217728, "aa6be8103f6c3a680f145c30dcc9c730dc1fa044d5389dc4627906a8531af83d"),
    (0, 939524096, 134217728, "df6187aafb5432542d60d369f14b375bb823ae53077c39383af71b67fe1e2511"),
    (0, 1073741824, 134217728, "e110076b6672e3ba3f98d1e6ad0d092546c2b9b06ec50af34867c05fd302f2dc"),
    (0, 1207959552, 134217728, "ce185e76667eb824185c5234fab117817909b669e490bb475314435557ffa790"),
    (0, 1342177280, 134217728, "1a61c746e64eb65242ec9a8a539962aa290e2210f054aa5c6504e12f2a3bc8d7"),
    (0, 1476395008, 134217728, "7bccb20d09ce3d29a1b1a60aae787903c749045897faa67435bbe577b24c195d"),
    (0, 1610612736, 134217728, "0780c4537de6fb2647b6ff15eb08e9b3847869cfc275a1ee889002a94ee14dab"),
    (0, 1744830464, 134217728, "19b0cc61b28901b3dd5082450bcd11ec88bcd1ab53f1a38a02cc58a0d55c18f3"),
    (0, 1879048192, 134217728, "a658f231f54163eb33e0d4c8fc74d0dc9952b17ed59d74d3e05a31d02dbede93"),
    (0, 2013265920, 134217728, "13c6753687f5e03718417ceef957086bb68e7dde9b5a6797ab0c2745fc24fa2d"),
    (0, 2147483648, 134217728, "572dfba84e08d3cad843ed5b576e3ea7fd17c3fdddd74c9a121b50c316b9c4b6"),
    (0, 2281701376, 134217728, "86244dad397ffff516eefc5ff12681fa6ccbc87f57e137d0e5c692d119f698a5"),
    (0, 2415919104, 134217728, "16319cd52348703ade16e1c8a0793c2974d54c77742c0107bf7f3247009e1341"),
    (0, 2550136832, 134217728, "4caa21ac4effcbfe2dceafd6e219c4de2402001ce1d5e861fcdffa56f9bb21e5"),
    (0, 2684354560, 134217728, "8926414007e79fbbafb636ae00cc1aa2a02a6088c5bd617a64eae8a5c16a8bd5"),
    (0, 2818572288, 134217728, "92af750544d75c2466f35d91b735de503b22ac4457959ef5c7d2633fef2223ee"),
    (0, 2952790016, 134217728, "c0ae650f0f2d02cc80d34f1c1d3c69ceddbf1fee211e7101f80bf2fc3e9fc143"),
    (0, 3087007744, 134217728, "2a9b6290cb7a366020952e8bf78914108540175d639b451a20fc55c63b2a8b35"),
    (0, 3221225472, 134217728, "c1fa7278d80c63c85947ce186db624a8969bbb94fbc91fee1f25ed19c6e2577c"),
    (0, 3355443200, 134217728, "800e32b165efcbb87c16edb39ad68c7f526136d77b822159c2178f3931d08e35"),
    (0, 3489660928, 134217728, "831be4ee791938012af5e503f67622e3fc61230da573ae777f36923183306baf"),
    (0, 3623878656, 134217728, "19c14f0f4e5048613a88a1cb0fe1a72b074b6c705b78e35606f17539c323a71c"),
    (0, 3758096384, 134217728, "140152fdce6b06750e0209e41f8f6283de582e8a3d6fc2bde2b4bf1ecbf5eb09"),
    (0, 3892314112, 134217728, "7f88b9f35df6f7e5d3fc48eaf85844d36cdd73e3d08eaea1c15452121c0c905f"),
    (0, 4026531840, 134217728, "a3fec4b0330dc19fb2d2a1a9ed859bda86c0bb22b8ffe592c444331bc88fdf64"),
    (0, 4160749568, 134217728, "f7a6b3be85750bbc0fe33f40389b39f5dba97f57bdf54d444cb779e88438da8f"),
    (0, 4294967296, 134217728, "76c0b29aa739b4fb2baf1fb50f07b214a8693811255d0c39bb0fc9f99c3b66ea"),
    (0, 4429185024, 134217728, "ac7439a2dc4edc87256c675dce454b5e4b1708ba0e1335d667b97c2cb3da0bea"),
    (0, 4563402752, 134217728, "b4c6ca6ed181b7daf7ab7436e1669d50db83b843ef207786342d1b8443e7bc15"),
    (0, 4697620480, 134217728, "7092470487b786f15f9d4ca032e85997c863fcd4d3e8071fc4a14f69116c48f0"),
    (0, 4831838208, 134217728, "511ae41770bede81031499512ca071e0b8323de66c54973299dcc3faf7dabdee"),
    (0, 4966055936, 134217728, "e53573a61c2eaa647d28f66c79fe7399cf7eb5be5c136284c17f402c8270d7ff"),
    (0, 5100273664, 134217728, "8a49eb031e5bbb73461f7b29e8c3ad0dde9623c80cf58c5fa4f4705052ad3777"),
    (0, 5234491392, 134217728, "c6786d099f9337fd1f2b2669fec06f15691c3ffeb084c9da19b25b2b5fa0b38d"),
    (0, 5368709120, 134217728, "1f7d30d2b841039188a13a3ef174c5a4263f9fa65a4fc19ff612cdd890f3fcec"),
    (0, 5502926848, 134217728, "95c726c689245beb8624d46047a06888c6441599a1cc6e08bab3aa2229412082"),
    (0, 5637144576, 134217728, "46721a8185d0e0f7bce523997543675bf5e65340550f76e58271c48c0fcc3209"),
    (0, 5771362304, 134217728, "86e8df9adfc27992db73ffbe2d22fd71bfb45dac96083619883cce34cb2d221a"),
    (0, 5905580032, 134217728, "8428075d8e84af3f6004d893bafda92e87b2ce82b020a1b0377b7455968c9bcd"),
    (0, 6039797760, 134217728, "fdafebb61ad128c72b236d41d4880f77bae801fb205a426c88d5cb1479221753"),
    (0, 6174015488, 134217728, "ca49ddb09e624267cb7d306829f65a60e7af32d4d622083ad7c25dc66c037f3e"),
    (0, 6308233216, 134217728, "6f4bf4c4eabdb0bbb82bde03696e488165d7c636c0d47be62b91f7e58fde6e7c"),
    (0, 6442450944, 134217728, "2f9e38db632d6d2acaf227af347f49502d185eb9f1e807fcf96212b7136c074a"),
    (0, 6576668672, 134217728, "5a092d6837d4bc5f9e31fa123ac10f59238096a5242b86634efbd54954131fbf"),
    (0, 6710886400, 134217728, "fb95b0ca9da87d8df4bb54b1e079af01ca26dfad185ed3bf4d3cf1ab5471c4b1"),
    (0, 6845104128, 134217728, "112975656e83b8a40396f3f028b26a22b5ecc62809ed0a877b3120301dc0e471"),
    (0, 6979321856, 134217728, "d11da41e9048dd0e75d479378e98e4a843da0e7e1a9cebda76688a709a2d6e33"),
    (0, 7113539584, 134217728, "23b97ff1ffe37498ad6529577164dda01494fad56eeab56f765e6ba2cf5491b6"),
    (0, 7247757312, 134217728, "e03db06b06b7b9e5fe2166c194d2a93a5604835c65ba9757b683a11268a41bbe"),
    (0, 7381975040, 134217728, "0d74a7572d60ec068c2b0516b46e576bf8af6caaf5719cbac5ed37200ac0d225"),
    (0, 7516192768, 134217728, "6069fdfc401a5f0c6ff9ba64af29876234c607c5eb242596704201b788abd96e"),
    (0, 7650410496, 134217728, "d77cf58842e62ef45a640b50dbe3aa9b9d9237abe1be09796aeefe5a412ce4f5"),
    (0, 7784628224, 134217728, "7405b6bd62f60b63b9e1079c8ea4c90b427f6d17e00219ac735b96a60db603e4"),
    (0, 7918845952, 134217728, "62e79d09b110abb839da647132df6df6d75a8b7bf9726f7db9416a026715ca4c"),
    (0, 8053063680, 134217728, "6cf5eb252c9f91a04ac7e5317db92d96dd7a812215dab3b9786db1f16703ed75"),
    (0, 8187281408, 134217728, "5702fefddfdee5bd228d66df1a8655654696a29387b55e22db9ffbded48e113b"),
    (0, 8321499136, 134217728, "14d088e9a2af60cef412a167a8e816a8c43efafc8b7f44917f9214cfb5857801"),
    (0, 8455716864, 134217728, "47c09f4cc56010c66c58197d35172b80ad65686fbbc005d35a3eb13ffea16cda"),
    (0, 8589934592, 134217728, "3aab2818bca4705cbe8e8ab4bf829b3e4c59d7e5da2a93c0a12240a6152d721d"),
    (0, 8724152320, 134217728, "912776ee9a9365fbf1ed8e8db21768b2d90f1f5148c46e88b101c13025af74a5"),
    (0, 8858370048, 134217728, "f4b2d24230572af48a467bd1a1c9e53aa8668f3b99bd4b96022c2fccc60d42bd"),
    (0, 8992587776, 134217728, "14c9c27db48712e977bd99b0132ea77242916dfbc02eb2f6f0c82aefb6cbff23"),
    (0, 9126805504, 134217728, "9ca1d7cac9e57d1dd7bff94d64956922cd57d8fefbc289569cfcc91bb071602e"),
    (0, 9261023232, 134217728, "d4cd683e3da877ea6a733ca9a4f5b970601343b93dab5570ba14e759a645ba9c"),
    (0, 9395240960, 134217728, "e3e14d90597c266dd619fb3dbeff49d00311dc41629327c5cd8b6ef5a8095fd5"),
    (0, 9529458688, 134217728, "63b154c1be43420dd81662ebb98476e0cc10bea67b91ec8b1f445fa8bbd3893c"),
    (0, 9663676416, 134217728, "c03efdc8376029f9040bb288aea495f9671b7afcfe0ec63b96e57ee7e7581455"),
    (0, 9797894144, 134217728, "620f44a0632abad2b42cb1830e2f7c64858771b3e2b8f770fb72d10a9f38812c"),
    (0, 9932111872, 44623547, "0af65ede4953f9c06a9c8a2bd41eff5ab2b19fa21a6c32246d2e36ee56e45a8d"),
    (1, 0, 134217728, "e34f5cdadef836510bdcca2e622190e7ed4628a735c773e36205be9dad3ae8a2"),
    (1, 134217728, 134217728, "60b753707daf1c9a96ac28e2f04575bc61df66509af23480b786263a2baca9bc"),
    (1, 268435456, 134217728, "b5a8b3249b6a79c92850eddc1b13360a19f53cae53835a4f9b9f3277bff47998"),
    (1, 402653184, 134217728, "a6f1fae8f100183ad7a3e575830c158455316febf88d06d3ccc5e7409380e926"),
    (1, 536870912, 134217728, "5b9acbde84fa7416658c39471557099fb21e76f55d2ccd5616e98aae1a2c2bfe"),
    (1, 671088640, 134217728, "65915b0c6f736f00799590adc91cc38741742588905efc020efb85db9f45de92"),
    (1, 805306368, 134217728, "a0e4d2823695c255c6cc08c41b6ce34239a9f30a7aa36e58fa70eff6201a0eb3"),
    (1, 939524096, 134217728, "54bacdfa04b140e9e2554cc3b33690ea6b0abbeb4c4c382b64b66ad98c733951"),
    (1, 1073741824, 134217728, "dd61ecff16cc4d5cc6b7c7825a1e09fca42e5b2e9ef931ab7d52a10552cd4e9d"),
    (1, 1207959552, 134217728, "f2486343a5b75772c095c36793c72e059ddd5675234625f879e14f79e96e6bf9"),
    (1, 1342177280, 48169775, "ea45c985d06d4b0756c4c48659c65f9a8f40e3eee796505294c56c24c99f77f5"),
]

# The flat chunk index each shard's parts start at: part%02d within its shard.
FIRST_CHUNK = {}
_next = 0
for _name, _size, _sha in SHARDS:
    FIRST_CHUNK[_name] = _next
    _next += sum(1 for row in CHUNKS if SHARDS[row[0]][0] == _name)

READ_TIMEOUT_S = 20    # a read that moves no bytes for this long is a stalled connection
STEP_DEADLINE_S = 240  # this step fails, having owed the builder only this much
BLOCK = 1 << 20


def log(message):
    print("[fetch_chunk %s] %s" % (time.strftime("%H:%M:%S"), message), file=sys.stderr, flush=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def note_source():
    with open(os.path.join(MODEL_DIR, ".weights_source"), "w") as f:
        f.write("facebook/esm2_t36_3B_UR50D @ %s\n" % REVISION)
        for name, size, sha in SHARDS:
            f.write("fp32 %s sha256 %s\n"
                    "shipped as verified 128 MiB chunks; the wrapper assembles them at load\n"
                    % (name, sha))


def fetch_once(name, url, start, length, want, target):
    """One ranged attempt from one source. True when target holds the pinned chunk."""
    have = os.path.getsize(target) if os.path.exists(target) else 0
    if have > length:
        log("%s: %d bytes held is past the chunk's %d; starting over" % (name, have, length))
        os.remove(target)
        have = 0
    end = start + length - 1
    headers = {"User-Agent": "brineylab-toolshed-esm2-3b-build",
               "Range": "bytes=%d-%d" % (start + have, end)}
    req = urllib.request.Request(url, headers=headers)
    began = time.monotonic()
    with urllib.request.urlopen(req, timeout=READ_TIMEOUT_S) as r:
        if r.status != 206:
            log("%s: a range request came back HTTP %s, not 206" % (name, r.status))
            return False
        content_range = r.headers.get("Content-Range", "")
        if not content_range.startswith("bytes %d-" % (start + have)):
            log("%s: served %r, not the range asked for" % (name, content_range))
            return False
        log("%s: chunk bytes %d-%d, %d already held, content-length %s"
            % (name, start + have, end, have, r.headers.get("Content-Length")))
        with open(target, "ab" if have else "wb") as f:
            while True:
                block = r.read(BLOCK)
                if not block:
                    break
                f.write(block)
                have += len(block)
    got = os.path.getsize(target)
    if got != length:
        log("%s: %d of %d bytes; the connection ended early" % (name, got, length))
        return False
    digest = sha256_of(target)
    if digest != want:
        log("%s: the chunk hashes to %s, not the pinned %s; refetching it from byte 0" % (name, digest, want))
        os.remove(target)
        return False
    log("%s: chunk delivered and verified in %.1f s" % (name, time.monotonic() - began))
    return True


def fetch(index):
    shard_index, start, length, want = CHUNKS[index]
    shard = SHARDS[shard_index][0]
    target = os.path.join(MODEL_DIR, "%s.part%02d" % (shard, index - FIRST_CHUNK[shard]))

    if os.path.exists(target):
        size = os.path.getsize(target)
        if size == length and sha256_of(target) == want:
            log("chunk %d already holds its pinned %d bytes; nothing to do" % (index, length))
            return True
        if size >= length:
            log("chunk %d's file holds all %d of its bytes but not the pinned ones; refetching it"
                % (index, size))
            os.remove(target)
        else:
            log("chunk %d's file holds %d of %d bytes; the next attempt resumes from there"
                % (index, size, length))

    os.makedirs(MODEL_DIR, exist_ok=True)
    deadline = time.monotonic() + STEP_DEADLINE_S
    attempt = 0
    while time.monotonic() < deadline:
        name, template = SOURCES[attempt % len(SOURCES)]
        attempt += 1
        try:
            if fetch_once(name, template % shard, start, length, want, target):
                if index == len(CHUNKS) - 1:
                    note_source()
                return True
        except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout,
                TimeoutError, ConnectionError, OSError) as e:
            log("%s: attempt %d failed: %s: %s" % (name, attempt, type(e).__name__, e))
    log("FATAL: chunk %d (%d bytes of %s at offset %d) not delivered in %d s"
        % (index, length, shard, start, STEP_DEADLINE_S))
    return False


def main():
    if len(sys.argv) != 2:
        log("usage: fetch_chunk.py <index 0-%d>" % (len(CHUNKS) - 1))
        return 2
    try:
        index = int(sys.argv[1])
    except ValueError:
        log("the chunk index is not a number: %r" % sys.argv[1])
        return 2
    if not 0 <= index < len(CHUNKS):
        log("chunk index %s is outside 0-%d" % (sys.argv[1], len(CHUNKS) - 1))
        return 2
    for shard_index, (name, size, sha) in enumerate(SHARDS):
        rows = [row for row in CHUNKS if row[0] == shard_index]
        if sum(row[2] for row in rows) != size:
            log("FATAL: %s's chunks hold %d bytes, not the pinned %d"
                % (name, sum(row[2] for row in rows), size))
            return 1
        offset = 0
        for _, start, length, _sha in rows:
            if start != offset:
                log("FATAL: %s's chunks are not contiguous at %d" % (name, start))
                return 1
            offset += length
    log("python %s; chunk %d of %d: %d bytes of %s at offset %d"
        % (sys.version.split()[0], index, len(CHUNKS), CHUNKS[index][2], SHARDS[CHUNKS[index][0]][0], CHUNKS[index][1]))
    return 0 if fetch(index) else 1


if __name__ == "__main__":
    sys.exit(main())
