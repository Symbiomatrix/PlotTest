# -*- coding: utf-8 -*-
#!! -*- coding: cp1255 -*-
"""
Created on 26/05/18

@author: Symbiomatrix

@purpose: Simple pyplot wrapper.
Maps parms to fig - axis - plot.
Inheritance through dict parm percolation;
 VERY IMPORTANT to add new parms to viewd.
Interactive plot in spyder: tools -> prefs -> ipython -> graphics -> backend -> qt.
In eclipse: window -> preferences -> Pydev -> Interactive Console ->
 -> Enable GUI event loop integration = pyqt5.

Todo:
- Zoom function using x/ylim (somewhat redundant with interactivity.)
- Update fig / ax parms - dict of parms triggering a function call + removal.

Bugs:
    Figsize calc is inaccurate at high widths.

Notes:
- The mechanism is in part built around spyder, wherein all files run
  on a single console - creating new figure instances floods the taskbar,
  so the module keeps global track of its spawned entities.
- In eclipse, there are 2 types of behaviour.
  (1) In console (input) mode, slightly different from spyder - figs show unblocked,
  display normally, die with the console (which can be swapped out for convenience);
  However, cannot update the same fig num with show (even under new object). 
  (2) In file run, a new process is created, wherein only plt.show(block = True)
  ostensibly works - without it, blank figs and automatic closure on end.
  However, draw + pause also does the job, better - plot gui is functional for the
  duration of the pause, draw nonblocking, and permits update. 
  Still must plot all at once (dunno what is fig.draw(artist)).
  To take advantage of this, use the input-display function:
  This waits for timed input of reset / quit, whilst granting a certain duration
  for fiddling with the graph (should be imperceptible).
- When altering the module, delete any global instances, or they retain old funcs.
  (Can be automated using version parm branding.)

Version log (incrementing on imports):
03/05/19 V3.0 Forked to eclipse.
06/06/18 V2.6 Improved manual grid row / col setting.
                Added approx rect (user calls).
                Improved locol settings - all / r / c / no matching.
                Added masking to square expansion.
02/06/18 V2.5 Added vmin, vmax parms to imshow.
                Grid sets these to global / local mat values by ind.
28/05/18 V2.4 Improvements for imshow:
                Added asymmetric axes through gspec parms, 
                hiding grid labels in axis function.
                Custom gspec for large matrix / multi mat display.
                Added subplot adjustment, tightlay (useless) func to fig.
                Automatic figsize based on subplot sizes.
28/05/18 V2.3 Added imshow + colorbar.
                Converted ax parms to subdict.
                Fixed figure naming scheme, allows strings.
                Added existence functions.
28/05/18 V2.2 Delfig accepts lists (delax is a tad more complicated matching).
                Fixed bugs in del of nonexistent fig / axis and partial del.
                Added refresh fig (closed by user twixt runs).
27/05/18 V2.1 Added varying dict keys, modded suptitle.
                Added holdon to plot.
27/05/18 V2.0 Converted most parms to dict based and defaulted.
27/05/18 V1.0 Wrote base structure functions.
26/05/18 V0.0 New.

"""

import sys
import logging
logger = logging.getLogger(__name__) # Separated logger, does not spam main.
if __name__ == '__main__':
#    logging.basicConfig(stream=sys.stderr,
#                    level=logging.ERROR, # Prints to the console.
#                    format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
    logging.basicConfig(filename="errlog.log",
                        level=logging.WARN, # Prints to a file. Should be utf. #.warn
                        format="%(asctime)s %(levelname)s:%(name)s:%(message)s") # Adds timestamp.
    logger.setLevel(logging.DEBUG)
SYSENC = "cp1255" # System default encoding for safe logging.
FRMTMLOG = "%y%m%d-%H%M%S" # Time format for logging.
PRTLIM = 1000 # When printing too long a message, will invoke automatic clear screen.
BADLOG = "Utilog-d{}.txt"

logger.debug("Start pplot module")

import matplotlib.pyplot as plt # v
import matplotlib.gridspec as gsp # v
import numpy as np # v
import numpy.ma as msk # v
import Utils_PT as uti # v
gcd = np.gcd
Default_Dict = uti.Default_Dict
View_Dict = uti.View_Dict
Parm_Consolidate = uti.Parm_Consolidate

# BM! Consts
Deb_Prtlog = lambda x,y = logging.ERROR:uti.Deb_Prtlog(x,y,logger)
ARRTYPE = np.array([]).__class__
# Proportion of figures.
A4SCALE = np.array([210,297]) / 25.4 # Kinda arbitrary values - full screen height and 2 / 3 screen width.
MAXSQUARE = np.array([7,6]) # The resulting sizes for a perfect square figure - includes edge margins, window wrap.
MSQWID = 2 * MAXSQUARE[0] / np.sum(MAXSQUARE)
MAXSCREEN = np.array([13.5,6]) # In 2 row taskbar.
MSCWID = MAXSCREEN[0] / MAXSCREEN[1]
PROPRANGE = (0.1,0.9)
FLSWAIT = 0.001
NOWAIT = 1
USRWAIT = 30
# Defs and viewds ought to contain all known parms.
# Size set according to fullscreen width approximately.
FDEFS = {
"Fig":{"ppr":1,"ppc":1,"figsize":(13,3),
        "supttl":"Figure {fign}","supx":0.5,"supy":1.03,"supfont":14}, # y = 0.98 overlaps.
"Axis":{"pptitle":"Graph","ppxlabel":"X axis","ppylabel":"Y axis","ppindleg":False,
        "ppindalbl":True,"ppindgrid":True,"ppindmgrid":True},
"Plot":{"label":"Function {fn}",
        "ppfign":1,"ppaxr":0,"ppaxc":0,"holdon":True},
"Imgs":{"ppindleg":True, # Legend merged for ease of access.
        "ppfign":1,"ppaxr":0,"ppaxc":0,"holdon":False,
        "ppindalbl":False,"ppindgrid":True,"ppindmgrid":False,"ppxlabel":"","ppylabel":""},
"Fgrid":{"ppfign":1,"ppr":-1,"ppc":-1, # Will override rc if dict set.
         "ppindleg":False,"cmap":"gray", # Cbars unreadable in medium grid. Greyscale preferable.
         "ppindalbl":False,"ppindgrid":False,"ppindmgrid":False, # Save more space on grid tags.
         "adjh":0.05,"adjw":0.05,"adjl":0.05,"adjr":0.95,"figsize":None, # Tighter layout.
         "vmin":None,"vmax":None,"ppindlocol":1, # Local / global / rc brightness.
         "heatrmin":None,"heatrmax":None,"heatcmin":None,"heatcmax":None},
"Null":None
}
# View dictionary. Sends parms relevant to function, through the percolation.
# PP parms do not need this since no atterr is thrown for these.
# Note that the keys differ from defs, which refer to the pp function.
VIEWD = {
"Fig":{"figsize"},
"Figb":{"ppr","ppc","figspec"}, # Backup only (used outside fig cre).
"Axis":{"fcnt","pptitle","ppxlabel""ppylabel","ppindleg"},
"Plot":{"label","lw"},
"Imgs":{"cmap","vmin","vmax"},
"Null":set()
}
VIEWD2 = { # Conversion dicts. Can work for all keys, but unnecessary dupe.
"Supt":{"supttl":"t","supx":"x","supy":"y","supfont":"fontsize"},
"Gspec":{"ppr":"nrows","ppc":"ncols",
         "gshgts":"height_ratios","gswids":"width_ratios"}, # Does not accept dict, but can be coerced.
"Tgtlay":{"tgtpad":"pad"}, # h, wpad. Does nothing - use adjust.
"Adjsub":{"adjl":"left","adjr":"right","adjb":"bottom","adjt":"top", # 0 - 1, edge margin.
          "adjh":"hspace","adjw":"wspace"}, # 0 - 1 (0.2), inner distance.
"Null":dict()
}

# Not for user meddling.
USRFIGMSG = "Figure display mode. W = wait, R = refresh figs, Q / C = soft quit."
USRFIGREG = "^([wrqc])(.*)"
PUFIG = [False,USRFIGMSG,uti.LSTPH,USRFIGREG,-1] # Time varies.

def isarr(v1):
    """Checks whether variable is array.
    
    Spam."""
    if isinstance(v1,ARRTYPE):
        return True
    else:
        return False

def Aid_Rectsq(l,rrct,crct):
    """Used in approx rect.
    
    Spam."""
    if rrct >= 0 and crct >= 0:
        sqrow = rrct
        sqcol = crct
    elif rrct < 0 and crct < 0:
        sqrow = int(np.ceil(np.sqrt(l)))
        sqcol = int(np.ceil(l / sqrow))
    else:
        v1 = max(rrct,crct)
        v2 = int(np.ceil(l / v1))
        if rrct >= 0:
            sqrow = v1
            sqcol = v2
        else:
            sqrow = v2
            sqcol = v1
    
    return (sqrow,sqcol)

def Approx_Sq(vdisp,rrct = -1, crct = -1,colfill = False):
    """Converts a vector / number to a roughly square matrix.
    
    Sqrt, round up, divide, round up, fill remainder with zeroes.
    Permits rect given row / col, fixing one side and completing other.
    Expansion (ie row first to limited length) can be done,
    through clever transpose or reshape of input.
    Masking only affects color and value displayed - images always behind plots."""
    if isarr(vdisp):
        vdisp = vdisp.reshape(-1)
        l = vdisp.size
        sqrow,sqcol = Aid_Rectsq(l,rrct,crct)
        hidetail = msk.zeros(sqrow * sqcol - l)
        hidetail.mask = True # Always white, so more clearly distinguishable.
        sqdisp = msk.concatenate((vdisp,hidetail))
        if not colfill:
            sqdisp = sqdisp.reshape(sqrow,sqcol)
        else:
            sqdisp = sqdisp.reshape(sqcol,sqrow).T
        
        return [sqdisp,sqrow,sqcol]
    else: # Numerical size only (of list).
        l = vdisp
        sqrow,sqcol = Aid_Rectsq(l,rrct,crct)
            
        return [None,sqrow,sqcol]

def mgcd(lnum):
    """Greatest common divisor (factor) for an array / list of numbers.
    
    Returns a single value - gcd of entire list.
    Eg, if one of these is prime and one of the rest isn't a product then gcd = 1.
    No known quick variant of euclidean method for list -
    np's gcd can apply to arrs with broadcasting, but not chain them."""
    vmg = lnum[0]
    #for i in range(lnum.size - 1):
    i = 1
    indstop = False
    while not indstop:
        vmg = gcd(vmg,lnum[i])
        
        i = i + 1
        if i >= lnum.size or vmg == 1:
            indstop = True
    
    return vmg

def mlcm(lnum):
    """Least common multiple for an array.
    
    Equal to n / gcd."""
    vmg = mgcd(lnum)
    return (lnum / vmg).astype(int)

class Multiplot():
    """Graph / image dictionary.
    
    May contain variable subplots and several graphs per figure."""
    
    def __init__(self):
        """Init.
        
        Spam"""
        self.dplt = dict()
        self.daxe = dict()
    
    def __getitem__(self,key):
        """Squaries.
        
        Spam"""
        return self.dplt[key]
    
    def Ext_Fig(self,f):
        """Check if fig exists.
        
        Has key which isn't null."""
        if self.dplt.get(f,None) is None:
            return False
        else:
            return True
    
    def Ext_Ax(self,f,axr,axc):
        """Check if axis exists.
        
        Has key which isn't null."""
        if self.daxe.get((f,axr,axc),None) is None:
            return False
        else:
            return True
    
    def Get_Fig(self,f,**pfig):
        """Create new figure.
        
        Known parms: figsize = 10,8 (standard?).
        Rows and cols for subplot can only be set once,
        or break the view, so makes sense to set them here.
        All relevant parms for recreation stored in cache."""
        rund = Default_Dict("Fig",pfig)
        rund["supttl"] = rund["supttl"].format(fign = f)
        
        # CONT: Gspec init here.
        if not self.Ext_Fig(f):
            fig = plt.figure(num = f,**View_Dict("Fig",rund)) # num = f provides name and reference capability.
            logger.debug("Created figure {}, num {}".format(f,fig.number))
            drecr = View_Dict("Fig",rund,False)
            drecr.update(View_Dict("Figb",rund,False))
            drecr.update(View_Dict("Supt",rund,False))
            drecr.update(View_Dict("Gspec",rund,False))
            drecr.update(View_Dict("Tgtlay",rund,False))
            drecr.update(View_Dict("Adjsub",rund,False))
            # Alt: subplots(gridspec_kw = {"width_ratios":[3,1]}), but that fills up immediately.
            drecr["figspec"] = gsp.GridSpec(**View_Dict("Gspec",rund))
            self.dplt[f] = (fig,drecr)
            # Visual aesthetics - upper title.
            fig.suptitle(**View_Dict("Supt",rund)) #,fontsize=18)
            d = View_Dict("Tgtlay",rund)
            if len(d) > 0:
                fig.tight_layout(**d)
            d = View_Dict("Adjsub",rund)
            if len(d) > 0:
                fig.subplots_adjust(**d)
        
        return self.dplt[f]
    
    def Refresh_Fig(self,f):
        """Reopen a figure closed by user.
        
        Show returns a lingering image, but is buggy and cannot update,
        so best to recreate the fig in that case.
        New: Draw seems to do the trick."""
        verr = 0
        if not self.Ext_Fig(f):
            verr = 2
        else:
            fig = self.Get_Fig(f)
            if plt.fignum_exists(fig[0].number):
                # Figure is still open.
                verr = 0
            else:
                # Figure was closed - cannot reregister, so create a new one.
                #fig
                #fig.show()
                verr = 1
                fig[0].canvas.draw_idle() # Idle = draw if not busy or summat.
                plt.pause(FLSWAIT)
                # Old method - destroy and create.
                #rund = fig[1]
                #self.Del_Fig(f,True) # Wipe the ref.
                #self.Get_Fig(f,**rund)
        
        return verr
    
    def Grid_Set(self,ax,**rund):
        """Grid related functions.
        
        For some reason, images refuse to be set the first time?"""
        verr = 0
        # To set label / grid en masse, loop over fig.axes.
        if rund["ppindmgrid"]: # Full grid overlay.
            ax.grid()
        # Set_ticks for just gridlines.
        ax.get_xaxis().set_visible(rund["ppindgrid"]) # Turns axis grid + labels.
        ax.get_yaxis().set_visible(rund["ppindgrid"])
        if not rund["ppindalbl"]: # Numerical axis label removal.
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
        
        return verr
    
    def Get_Ax(self,f,axr,axc,**paxe):
        """Create subplot for drawing.
        
        Title, x and y labels can be altered, but best preserved.
        Can convert gspec notation to subplot, but it's kinda shippy.
        Creep: If asymmetric grid is required, use gs indexing eg [1:,2],
        and convert axr+c to tups whose first elem used in d/axe."""
        
        rund = Default_Dict("Axis",paxe)
        
        if not self.Ext_Ax(f,axr,axc):
            fig = self.Get_Fig(f,**rund)
            #r = fig[1]["ppr"] # Base method.
            #c = fig[1]["ppc"]
            #ax = fig[0].add_subplot(r,c,i)
            ax = fig[0].add_subplot(fig[1]["figspec"][axr,axc])
            ax.set_title(rund["pptitle"])
            ax.set_xlabel(rund["ppxlabel"])
            ax.set_ylabel(rund["ppylabel"])
            self.Grid_Set(ax,**rund)
            drecr = {"fcnt":0}
            drecr.update(View_Dict("Axis",rund,False))
            # if islst(axr): axr[0]
            self.daxe[(f,axr,axc)] = (ax,drecr)
        
        return self.daxe[(f,axr,axc)]
    
    def Plot(self,xvals,yvals,fign = None,axr = None,axc = None,**pgraph):
        """Plot a 2d function.
        
        Uses a quick setup - to add container parms, call get fig / ax directly.
        Known parms: lw = linewidth, label (in legend).
        Other graph related functions activated on ax creation,
        bar legend which must be called after each plot."""
        verr = 0
        
        rund = Default_Dict("Plot",pgraph)
        
        fign = Parm_Consolidate("ppfign",fign,rund)
        axr = Parm_Consolidate("ppaxr",axr,rund)
        axc = Parm_Consolidate("ppaxc",axc,rund)
        
        self.Refresh_Fig(fign) # Ensure the plot isn't wasted.
        ax = self.Get_Ax(fign,axr,axc,**rund)
        ax[1]["fcnt"] += 1 # Number of functions on graph.
        rund["label"] = rund["label"].format(fn = ax[1]["fcnt"])
        
        if not rund["holdon"]:
            self.Del_Ax(fign,axr,axc)
            self.Grid_Set(ax[0],**rund)
        ax[0].plot(xvals,yvals,**View_Dict("Plot",rund))
        if ax[1]["ppindleg"]:
            ax[0].legend()
#            fig = self.Get_Fig(fign,**rund)
#            fig[0].legend() # Mayhap less convenient normally, and doesn't work in image overlays.
        #fig.show() # Inline gives an odd warning about gui not configured.
        
        return verr
    
    def Imshow(self,mimg,fign = None,axr = None,axc = None,**pgraph):
        """Display an image in pixel form.
        
        Only one image per axis (automatic overlapping, till figuring it out),
        so it's a variable rather than l3 dict.
        Known parms: cmap "gray" for greyscale, "hot" bryw.
        Converting an image to grayscale:
        from PIL import Image
        fname = 'image.png'
        image = Image.open(fname).convert("L")
        arr = np.asarray(image)"""
        verr = 0
        
        rund = Default_Dict("Imgs",pgraph)
        
        fign = Parm_Consolidate("ppfign",fign,rund)
        axr = Parm_Consolidate("ppaxr",axr,rund)
        axc = Parm_Consolidate("ppaxc",axc,rund)
        
        self.Refresh_Fig(fign) # Ensure the plot isn't wasted.
        self.Del_Ax(fign,axr,axc)
        ax = self.Get_Ax(fign,axr,axc,**rund)
        ax[1]["fcnt"] += 1 # Number of functions on graph.
        
        #if not rund["holdon"]: # Cannot display more than one. Barely a plot over image.
        self.Del_Ax(fign,axr,axc)
        self.Grid_Set(ax[0],**rund) # Reset the grid.
        im = ax[0].imshow(mimg,**View_Dict("Imgs",rund))
        ax[1]["ppimg"] = im
        # Applying bar to ax steals space from the image itself,
        # Fig is sometimes (on err?) on the furthest right of ALL axes.
        if ax[1]["ppindleg"]:
            fig = self.Get_Fig(fign,**rund)
            fig[0].sca(ax[0]) # Puts colorbar next to current axis.
            fig[0].colorbar(im) # No need to save, can be retrieve.
        #fig.show() # Inline gives an odd warning about gui not configured.
        
        return verr
    
    def Fig_Prop(self,maxwids,maxhgts):
        """Estimates size of figure for proportional pixel display.
        
        Does not take edge / inner margin into account directly,
        only as square being slightly uneven.
        Uses the raw wids / hgts - no lcm scaling!"""
        ttlwids = np.sum(maxwids)
        ttlhgts = np.sum(maxhgts)
        # Can add a little bit for inner margin using len, insignificant.
        widprop = ttlwids / (ttlwids + ttlhgts)
        widprop = widprop * MSQWID
        widprop = min(max(widprop,PROPRANGE[0]),PROPRANGE[1])
        
        # Formula explained: to match perfect square aspect with skewed square, calc 1 / (1 + 1) * c = 7 / (7 + 6).
        # Then, nw / (nw + nh) = r => nw / nh = r / (1 - r).
        # One axis is set to max and the other *reduced* in accordance.
        mdfy = widprop / (1 - widprop)
        figsize = MAXSCREEN.copy()
        if mdfy > MSCWID: # Shrink width rather than increase height.
            figsize[1] = figsize[0] / mdfy
        else:
            figsize[0] = figsize[1] * mdfy
        
        return (figsize,widprop,1 - widprop)
    
    def Plan_Grid(self,submats,fign = None,indspl = -1,autodisp = True,**pgrid):
        """Create grid for a given list of matrices OR single large matrix to split.
        
        Assigns these to a roughly square area by def,
        picks the longest / widest arr in each row / col respectively,
        runs lcm to calc minimal ratios of visually similar pixel sizes.
        It would be more compact to sort by height globally then width per row,
        however order might be expected for the user so that's up to them.
        Most compact is probably multi index grid (of singletons),
        which is completely asymmetric in view.
        Split setting takes a single matrix, and transmutes its dimension
        to the subplot (list) size.
        If set to autodisp, will call imshow for each submat with the same dict parms.
        Otherwise, the stretched and shaped list is returned for single calls.
        When color set to local, each subplot will show a non comparable max,
        visually confusing unless intended."""
        verr = 0
        
        rund = Default_Dict("Fgrid",pgrid)
        
        fign = Parm_Consolidate("ppfign",fign,rund)
        
        if indspl >= 0:
            # Only difference twixt roll and move seems to be when moving right;
            # Move will displace dest idx to left, obtaining its idx, roll (always) to right.
            # Move to front makes loop eliminate specific axis whilst leaving others intact.
            smroll = np.moveaxis(submats,indspl,0)
            submats = [sm for sm in smroll] # Lcomp kills the new ax0 automatically.
            sqsub = [Approx_Sq(sm)[0] for sm in submats]
        else:
            if isarr(submats):
                sqsub = [submats] # One big mat - same as calling imshow directly.
            else: # List.
                sqsub = submats # Displayed as they are, potentially inner mats uneven.
        
        (_,rund["ppr"],rund["ppc"]) = Approx_Sq(len(sqsub),rund["ppr"],rund["ppc"])
        grrow = rund["ppr"]
        grcol = rund["ppc"]
        
        # Add extra zero mats to match row / col.
        grlen = len(sqsub) # Real images only displayed.
        if len(sqsub) < grrow * grcol:
            sqsub.extend([np.zeros((1,1))] * (grrow * grcol - grlen))
            
        # Split wid and hgt as evenly as possible.
        mdims = np.array([sm.shape for sm in sqsub])
        mdims = mdims.reshape(grrow,grcol,2) # Each mat shape positioned in grid.
        maxwids = np.max(mdims,0)[:,1] # Max for each col, take wid parm.
        maxhgts = np.max(mdims,1)[:,0] # Max for each row, take hgt parm.
        maxwids2 = mlcm(maxwids) # Remove any divisible factors (gcd), eg 2:4 = 1:2.
        maxhgts2 = mlcm(maxhgts)
        if rund.get("gswids",None) is None:
            rund["gswids"] = maxwids2
        if rund.get("gshgts",None) is None:
            rund["gshgts"] = maxhgts2
        if rund.get("figsize",None) is None:
            (figsize,_,_) = self.Fig_Prop(maxwids,maxhgts)
            rund["figsize"] = figsize
        
        heatrng = (np.array([np.min(sm) for sm in sqsub]).reshape(grrow,grcol),
                   np.array([np.max(sm) for sm in sqsub]).reshape(grrow,grcol))
        if rund["ppindlocol"] == 0: # Local.
            pass # Or set per mat, does the same thing as null vmin / vmax.
        if rund["ppindlocol"] == 1: # Global.
            if rund["vmin"] is None:
                rund["vmin"] = np.min(heatrng[0])
            if rund["vmax"] is None:
                rund["vmax"] = np.max(heatrng[1])
        if rund["ppindlocol"] == 2: # Rows.
            if rund["vmin"] is None and rund["heatrmin"] is None:
                rund["heatrmin"] = np.min(heatrng[0],axis = 1)
            if rund["vmax"] is None and rund["heatrmax"] is None:
                rund["heatrmax"] = np.max(heatrng[1],axis = 1)
        if rund["ppindlocol"] == 3: # Cols.
            if rund["vmin"] is None and rund["heatcmin"] is None:
                rund["heatcmin"] = np.min(heatrng[0],axis = 0)
            if rund["vmax"] is None and rund["heatcmax"] is None:
                rund["heatcmax"] = np.max(heatrng[1],axis = 0)
            
        #else vmin = None, vmax = None - or set per imshow call.
        
        self.Get_Fig(fign,**rund)
        if autodisp:
            for i in range(grrow):
                for j in range(grcol):
                    if i * grcol + j < grlen:
                        if rund["heatrmin"] is not None: # Color ranges.
                            rund["vmin"] = rund["heatrmin"][i]
                        if rund["heatrmax"] is not None:
                            rund["vmax"] = rund["heatrmax"][i]
                        if rund["heatcmin"] is not None:
                            rund["vmin"] = rund["heatcmin"][j]
                        if rund["heatcmax"] is not None:
                            rund["vmax"] = rund["heatcmax"][j]
                        verr = self.Imshow(sqsub[i * grcol + j],fign,i,j,**rund)
        
        return [verr,sqsub] # grrow,grcol can be extracted from fig dict.
    
    def Del_Ax(self,f,r,c,indkill = False):
        """Delete subplot.
        
        No known method for redistributing sp - 
        might silently copy to a new fig, leaving it manual for now.
        Does not delete since it messes up indices, loops."""
        verr = 0
        if self.Ext_Ax(f,r,c): # Prevent unnecessary creation
            ax = self.Get_Ax(f,r,c)
            ax[0].cla()
            ax[1]["fcnt"] = 0
            if ax[1].get("ppimg",None) is not None: # Ext img.
                cb = ax[1]["ppimg"].colorbar # Kill colorbars in sight.
                if cb is not None:
                    cb.remove()
            if indkill:
                self.daxe[(f,r,c)] = None
                #del self.daxe[(f,i)]
        
        return verr
        
    def Del_Fig(self,f,indkill = False):
        """Delete figure.
        
        Clears axes but leaves ref (and fig call parms) intact.
        For a complete reset set indication.
        Plt's close allegedly frees resources. Notwithstanding, show on an intact fig
        provides a lingering image, however it's buggy and cannot be registered back.
        As such, make sure to recreate the fig completely after nuking."""
        verr = 0
        if uti.islstup(f):
            f2 = f
        else:
            f2 = [f]
        for i in f2:
            if self.Ext_Fig(i): # Prevent unnecessary creation.
                for k in self.daxe: # Delete axes first (clf unbinds them).
                    if i == k[0]:
                        verr = self.Del_Ax(k[0],k[1],k[2],True)
                fig = self.Get_Fig(i)
                fig[0].clf()
                if indkill:
                    plt.close(fig[0].number)
                    self.dplt[i] = None # Cannot del fig var, that's a separate ref.
                    
        
        return verr
    
    def Passive_Display(self):
        """Opens all figures and listens for user commands.
        
        Adjusted for eclipse file processing mode:
        Does not block, permitting redraw and proceeding with the code.
        W {num} = Wait num secs until next waking.
        R {[figs] | all} = Redraw specific or all figs (comma / space sep).
        Q / C = Soft close (continue until reaching the proc's end, killing figs).
        Creep: Canvas draw seems to have no effect; pause redraws all rather than cur.
        Annoyingly, it also sets the focus on the entire series.
        As such, currently setting the duration higher.
        Creep: Reset the program entirely (rerun main), keeping the mplot refs.
        Currently, terminate + run doesn't seem a terrible inconvenience."""
        for fig in self.dplt.values():
            fig[0].show() # Alt: plt.show.
            
        indstop = False
        cslp = USRWAIT
        while not indstop:
            plt.pause(max(cslp - NOWAIT,NOWAIT)) # Only pause unlocks the gui.
            pwait = uti.List_Format(PUFIG,uti.LSTPH,NOWAIT)
            ursp = uti.Timed_Input(*pwait)
            
            if ursp:
                ursp2 = ursp.split(" ")
                comm = ursp2[0].lower()
                ursp2 = ursp2[1:]
                if comm.startswith("w"): # Change the wait schedule.
                    try:
                        cslp = int(ursp2[0])
                    except (ValueError,IndexError):
                        print("Cannot update period, still at {}.".format(cslp))
                elif comm.startswith("r"): # Redraw.
                    for fig in ursp2:
                        try:
                            fign = int(fig) # Numerical figs are stored as such.
                        except ValueError:
                            fign = fig
                        if fign in self.dplt:
                            self.dplt[fign][0].canvas.draw()
                        else:
                            print("Fig doesn't exist, cannot ref: {}.".format(fign))
                elif (comm.startswith("q") or comm.startswith("c")):
                    indstop = True 

# Checks whether imported.
if __name__ == '__main__':
    global mp # Necessary to overrun figs twixt runs.
    try:
        if mp is None:
            mp = Multiplot()
    except NameError:
        mp = Multiplot()
        
    tstx1 = np.arange(0,4,0.1).reshape(-1,1)
    tsty1 = np.sin(tstx1)
    tsty2 = np.cos(tstx1)
    tsty3 = np.tanh(tstx1)
    tsty4 = np.random.rand(*tstx1.shape)
    tsty5 = np.random.rand(*tstx1.shape)
    tstimg = [[1,0,0,0,1,1,1,0,1,0,0],[1,0,0,0,1,0,1,0,1,0,0],[1,1,1,0,1,1,1,0,1,1,1]]
    bigmat = np.random.rand(16,25)
    bigmat2 = np.random.rand(25,10)
    bigmats = [np.random.rand(i + 1,j + 2) for i in range(3) for j in range(3)]
    bigmats2 = [np.random.rand(i + j + 1,j + 2) for i in range(3) for j in range(3)]
    bigmats3 = [np.random.rand(i + 1,i + j + 2) for i in range(3) for j in range(3)]
    bigmats4 = [np.random.rand(2,8) for i in range(5) for j in range(5)]
#    mp = Multiplot()
    mp.Plot(tstx1,tsty1,**{"supttl":"Use your imagination."})
    mp.Plot(tstx1,tsty2)
#    mp.Get_Fig(2,{"ppr":1,"ppc":2}) # Redundant - dict percolation.
#    mp.Get_Ax(2,2,{"ppindleg":True})
    mp.Del_Fig(2,True)
    dsetng = {"ppr":1,"ppc":2,"ppindleg":True,"label":"Hobbit {fn}","holdon":True,"ppindalbl":False}
    mp.Plot(tstx1,tsty3,2,0,0,**dsetng) # Number of rows is real, but ax is 0 index.
    mp.Plot(tstx1,tsty4,2,0,1,**dsetng)
    mp.Plot(tstx1,tsty5,2,0,1,**dsetng)
    dsetng = {"ppr":2,"ppc":2,"ppindleg":True,"label":"Dwarf {fn}","holdon":True,"cmap":"gray","ppindgrid":False}
    mp.Imshow(tstimg,3,0,0,**dsetng)
    mp.Plot(tstx1,tsty3,3,0,0,**dsetng)
    mp.Plot(tstx1,tsty1,3,1,1,**dsetng)
    mp.Plot(tstx1,tsty2,3,1,1,**dsetng)
    dsetng["ppindgrid"] = True
    dsetng["ppindmgrid"] = True
    mp.Plot(tstx1,tsty4,3,1,0,**dsetng)
    mp.Plot(tstx1,tsty5,3,0,1,**dsetng)
    mp.Plan_Grid(bigmat,4,1)
    mp.Plan_Grid(bigmat2,5,1)
    mp.Plan_Grid(bigmats,6,**{"ppindlocol":2})
    mp.Plan_Grid(bigmats2,7)
    mp.Plan_Grid(bigmats3,8) # Superwide.
    mp.Plan_Grid(bigmats4,9)
    # Should work just fine with named figures.
    mp.Plot(tstx1,tsty2,"Applesauce",0,0,**dsetng)
    mp.Del_Ax(1,0,0)
    mp.Passive_Display()
    print("\nFin")
else:
    pass
    
# FIN
