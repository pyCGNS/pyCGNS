#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source
#  -------------------------------------------------------------------------
#
import CGNS.PAT.cgnsutils as CGU
import CGNS.PAT.cgnskeywords as CGK
import string

CHECK_NONE = 0
CHECK_OK = CHECK_GOOD = CHECK_PASS = 1
CHECK_INFO = CHECK_WARN = 2
CHECK_BAD = CHECK_ERROR = CHECK_FAIL = 3
CHECK_USER = 4


# def sortDiagByKey(d1, d2):
#     (k1, k2) = (d1.key, d2.key)
#     return k1 > k2


def getWorst(st1, st2):
    if CHECK_FAIL in [st1, st2]:
        return CHECK_FAIL
    if CHECK_USER in [st1, st2]:
        return CHECK_USER
    if CHECK_WARN in [st1, st2]:
        return CHECK_WARN
    if CHECK_GOOD in [st1, st2]:
        return CHECK_GOOD
    return CHECK_NONE


class DiagnosticMessagePattern(object):
    __levelstr = {
        CHECK_FAIL: "E",
        CHECK_USER: "U",
        CHECK_WARN: "W",
        CHECK_GOOD: " ",
        CHECK_NONE: "?",
    }

    def __init__(self, mkey, mlevel, mstring):
        self._key = mkey
        self._lvl = mlevel
        self._str = mstring

    @property
    def key(self):
        return self._key

    @property
    def level(self):
        return self._lvl

    @property
    def message(self):
        return self._str

    def levelAsStr(self):
        return self.__levelstr[self._lvl]

    def __str__(self):
        return "[%s:%s] %s" % (self._key, self.levelAsStr(), self._str)

    def notSubst(self):
        return "%" in self._str

    def forceLevel(self, l):
        self._lvl = l


class DiagnosticMessageInstance(DiagnosticMessagePattern):
    def __init__(self, pattern):
        super(DiagnosticMessageInstance, self).__init__(
            pattern.key, pattern.level, pattern.message
        )

    def substitute(self, *tp):
        msg = self._str
        try:
            if tp:
                msg = msg % tp
        except TypeError:
            pass
        self._str = msg
        return self

    def __str__(self):
        return '("%s","%s","""%s""")' % (self._key, self.levelAsStr(), self._str)


class DiagnosticLog(dict):
    __messages = {}

    def __init__(self):
        dict.__init__(self)

    def merge(self, log):
        self.update(log)

    def listMessages(self):
        return self.__messages

    def noContextMessage(self, m):
        if DiagnosticLog.__messages[m].notSubst():
            return None
        return DiagnosticLog.__messages[m].message

    def addMessage(self, k, m):
        DiagnosticLog.__messages[k] = DiagnosticMessageInstance(*m)

    def addMessages(self, d):
        for e in d:
            DiagnosticLog.__messages[e] = DiagnosticMessagePattern(e, d[e][0], d[e][1])

    def push(self, path, messagekey, *tp):
        if path is None:
            return
        if path not in self:
            self[path] = []
        if messagekey not in DiagnosticLog.__messages:
            return
        entry = DiagnosticMessageInstance(DiagnosticLog.__messages[messagekey])
        self[path].append(entry.substitute(*tp))
        return DiagnosticLog.__messages[messagekey].level

    def __len__(self):
        return len(self.keys())

    def shift(self, path, shiftstring=" "):
        n = path.split("/")
        return len(n) * shiftstring

    def getWorst(self, st1, st2):
        return getWorst(st1, st2)

    def status(self, entry):
        return entry.level

    def hasOnlyKey(self, path, keylist):
        k1 = set(keylist)
        k2 = set([e.key for e in self[path]])
        return k2.issubset(k1)

    def key(self, entry):
        return entry.key

    def message(self, entry, path=None):
        shft = ""
        if path is not None:
            shft = self.shift(path)
        s = "%s[%s:%s] %s" % (shft, entry.key, entry.levelAsStr(), entry.message)
        return s

    def getWorstDiag(self, path):
        s = set([e.level for e in self[path]])
        try:
            r = list(s)[0]
            for x in s:
                r = getWorst(r, x)
        except:
            r = CHECK_NONE
        return r

    def diagForPath(self, path):
        if path in self:
            return self[path]
        return None

    def allMessageKeys(self):
        r = set()
        for path in self:
            for entry in self[path]:
                r.add(entry.key)
        mlist = list(r)
        mlist.sort()
        return mlist

    def allPathKeys(self):
        return list(self)

    def diagnosticsByPath(self, path):
        if self.diagForPath(path) is not None:
            dlist = self[path]
            dlist.sort(key=lambda x: x.key)
            for diag in dlist:
                yield (diag, path)

    def diagnosticsByMessage(self, msg):
        plist = list(self)
        plist.sort()
        for path in plist:
            for diag in self[path]:
                if diag.key == msg:
                    yield (diag, path)

    def __str__(self):
        s = "{\n"
        for path in self:
            s += "'%s':\n" % path
            for diag in self[path]:
                s += "  %s,\n" % diag
        s += "}\n"
        return s

    def forceAsWarning(self, key):
        if key in DiagnosticLog.__messages:
            DiagnosticLog.__messages[key].forceLevel(CHECK_WARN)

    def forceAsFailure(self, key):
        if key in DiagnosticLog.__messages:
            DiagnosticLog.__messages[key].forceLevel(CHECK_FAIL)


# --- last line
